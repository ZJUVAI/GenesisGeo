"""Minimal Singular subprocess protocol for exact guard search."""

from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

_PROGRAM = r"""
ring r=0,(@VARS@),dp;
short=0;
ideal I=@INPUTS@;
ideal G=@INITIAL_BASIS@;
attrib(G,"isSB",1);
ideal initial_basis=G;
ideal F=@FACTORS@;
poly h=@GOAL@;
intvec used; intvec powers;
int nused=0; int inconsistent=0; int i; int j; int k; int calls=0; int feasible;
list L; ideal H; poly f;
if(reduce(1,G)==0) {inconsistent=1;}
for(i=1;i<=@NREQ@;i++) {
 f=reduce(F[i],G);
 if(f==0) {inconsistent=1;break;}
}
// A direct localization identity already proves the implication. Keep the
// original basis and record its nonzero multiplier for the FLINT verifier.
poly hrem=reduce(h,G);
poly h2rem;
if(hrem!=0) {
 if(size(h)<size(hrem)) {h2rem=reduce(h^2,G);}
 else {h2rem=reduce(hrem^2,G);}
}
poly once; poly twice; poly f2rem;
ideal reduced_factors;
int direct_exponent=0;
if(inconsistent==0 && hrem!=0 && h2rem!=0) {
 for(i=1;i<=@NFACT@;i++) {
  f=reduce(F[i],G);
  reduced_factors[i]=f;
  if(f!=0 && deg(f)>0) {
   once=reduce(hrem*f,G);
   if(once==0) {direct_exponent=1;k=1;}
   else {
    twice=reduce(once*f,G);
    if(twice==0) {direct_exponent=1;k=2;}
    else {
     once=reduce(h2rem*f,G);
     if(once==0) {direct_exponent=2;k=1;}
     else {if(reduce(once*f,G)==0) {direct_exponent=2;k=2;}}
    }
   }
   if(direct_exponent!=0) {
    nused++;used[nused]=i;powers[nused]=k;
    break;
   }
  }
 }
}
if(inconsistent==0 && direct_exponent==0 && hrem!=0 && h2rem!=0) {
 LIB "elim.lib";
 int basis_changed=0;
 for(i=1;i<=@NFACT@;i++) {
  if(basis_changed) {f=reduce(F[i],G);}
  else {f=reduced_factors[i];}
  if(f!=0 && deg(f)>0) {
   f2rem=reduce(f^2,G);
   if(f2rem==0 || deg(f2rem)>0) {
    calls++;
    if(calls>@MAXSAT@) {break;}
    L=sat(G,ideal(f));H=L[1];k=L[2];feasible=1;
    if(reduce(1,H)==0 || size(reduce(H,G))==0) {feasible=0;}
    for(j=1;j<=@NREQ@;j++) {if(reduce(F[j],H)==0) {feasible=0;}}
    if(feasible) {
     G=H;nused++;used[nused]=i;powers[nused]=k;
     basis_changed=1;
     hrem=reduce(h,G);
     if(hrem==0) {break;}
     h2rem=reduce(h^2,G);
     if(h2rem==0) {break;}
    }
   }
  }
 }
}
if(inconsistent) {print("NDG_STATUS:inconsistent");}
else {
 if(direct_exponent==0 && hrem!=0 && h2rem!=0) {print("NDG_STATUS:unproved");}
 else {
  int exponent=1;
  if(direct_exponent!=0) {exponent=direct_exponent;}
  else {if(hrem!=0) {exponent=2;}}
  print("NDG_STATUS:certified");
  print("NDG_USED:"+string(used));print("NDG_POWERS:"+string(powers));
  print("NDG_EXPONENT:"+string(exponent));
  if(@EMIT_BASIS@) {
   print("NDG_GB_SIZE:"+string(ncols(initial_basis)));
   for(i=1;i<=ncols(initial_basis);i++) {print("NDG_GB_"+string(i)+":"+string(initial_basis[i]));}
  }
 }
}
print("NDG_DONE");quit;

"""


@lru_cache(maxsize=1)
def _command():
    prefix = os.environ.get("DISCOVERY_EXPERIMENTAL_SINGULAR_PREFIX")
    home = (
        Path(prefix).expanduser() if prefix else Path(sys.prefix) / "share/ndg-singular"
    )
    packaged = home / "usr/bin/Singular"
    env = os.environ.copy()
    if packaged.is_file():
        binary = str(packaged)
        libraries = home / "usr/lib/x86_64-linux-gnu"
        env["LD_LIBRARY_PATH"] = f"{libraries}:{env.get('LD_LIBRARY_PATH', '')}"
        env["SINGULARPATH"] = str(home / "usr/share/singular/LIB")
        env["SINGULAR_PROCS_DIR"] = str(libraries / "singular/MOD")
    elif prefix:
        raise FileNotFoundError(f"Singular executable missing at {packaged}")
    else:
        binary = shutil.which("Singular")
        if binary is None:
            raise FileNotFoundError(
                "Install Singular or set DISCOVERY_EXPERIMENTAL_SINGULAR_PREFIX"
            )
    return binary, env


def search(inputs, variables, target, factors, required_count, basis, seconds):
    def polynomial_text(expression):
        return str(expression).replace("**", "^")

    values = {
        "VARS": ",".join(map(str, variables)),
        "INPUTS": ",".join(map(polynomial_text, inputs)),
        "INITIAL_BASIS": (
            ",".join(map(polynomial_text, basis)) if basis else "slimgb(I)"
        ),
        "FACTORS": ",".join(map(polynomial_text, factors)) or "1",
        "GOAL": polynomial_text(target),
        "NREQ": str(required_count),
        "NFACT": str(len(factors)),
        "MAXSAT": "22",
        "EMIT_BASIS": "0" if basis else "1",
    }
    source = _PROGRAM
    for name, value in values.items():
        source = source.replace(f"@{name}@", value)
    binary, env = _command()
    command = [
        "timeout",
        "--foreground",
        "--kill-after=1",
        str(seconds),
        binary,
        "-q",
        "--no-rc",
    ]
    try:
        process = subprocess.run(
            command,
            input=source,
            text=True,
            capture_output=True,
            env=env,
            timeout=seconds + 2,
        )
    except subprocess.TimeoutExpired:
        return "backend_timeout", None
    if process.returncode in {124, 137, -15}:
        return "backend_timeout", None
    if process.returncode or "NDG_DONE" not in process.stdout:
        detail = process.stderr.strip() or process.stdout.strip()[-500:]
        raise RuntimeError(f"Singular failed ({process.returncode}): {detail}")
    fields = dict(re.findall(r"^NDG_([A-Z_0-9]+):(.*)$", process.stdout, re.MULTILINE))
    verdict = fields.get("STATUS")
    if verdict == "unproved":
        return "unproved", None
    if verdict == "inconsistent":
        return "inconsistent_premises", None
    if verdict != "certified":
        raise RuntimeError(f"Unexpected Singular verdict: {verdict!r}")
    return "certified", fields
