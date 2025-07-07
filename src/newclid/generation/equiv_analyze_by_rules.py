import json
import sys
import os
import time
import random

class GeometryEquivalenceAnalyzer:

    def __init__(self, input_file):
        self.input_file = input_file

    def get_dataset_path(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "dataset", filename)

    @staticmethod
    def get_rules_from_proof(proof):
        """Extract rules from the proof string."""
        rules = []
        parts = proof.split('=>')[:-1]

        for part in parts:
            bracket_counter = 0
            i = len(part) - 1
            while i >= 0:
                if part[i] == ')':
                    bracket_counter += 1
                elif part[i] == '(':
                    bracket_counter -= 1
                if bracket_counter == 0:
                    break
                i -= 1
            rule = part[i:].strip()
            if rule:
                rules.append(rule)
        
        return rules

    @staticmethod
    def record_deduplicate_information(output_file, total_rows, rows_to_be_kept, time_taken):
        """Record deduplication information to an analysis file."""
        print(f"Total rows: {total_rows}, Rows to be kept: {len(rows_to_be_kept)}")
        print(f"Ratio of kept rows: {len(rows_to_be_kept) / total_rows * 100:.2f}%")
        print(f"Deduplicated file written to: dataset/{output_file}")
        print(f"Time taken: {time_taken:.2f} seconds")

        analysis_file = 'analysis_' + output_file.split('.')[0] + '_output.txt'
        with open(analysis_file, 'w', encoding='utf-8') as analysis_outfile:
            analysis_outfile.write(f"Total rows: {total_rows}\n")
            analysis_outfile.write(f"Rows to be kept: {len(rows_to_be_kept)}\n")
            analysis_outfile.write(f"Ratio of kept rows: {len(rows_to_be_kept) / total_rows * 100:.2f}%\n")
            analysis_outfile.write(f"Time taken: {time_taken:.2f} seconds\n")

    def deduplicate_by_rules(self, sample_num):
        print(f"Deduplicating {self.input_file} by rules...")

        start_time = time.time()
        id_of_proofs = {}
        total_rows = 0
        rows_to_be_kept = []
        output_file = 'deduplicated_by_rules_' + self.input_file
        output_file = output_file.split('.')[0] + f'_{sample_num}.jsonl'
        
        with open(self.get_dataset_path(self.input_file), 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                total_rows += 1
                data = json.loads(line.strip())
                llm_output = data['llm_output']
                proof = llm_output.split('<proof>')[1].split('</proof>')[0].strip()
                used_rules = self.get_rules_from_proof(proof)
                used_rules_string = ','.join(used_rules)
                # print(i, used_rules_string)
                if used_rules_string not in id_of_proofs:
                    id_of_proofs[used_rules_string] = [i]
                else:
                    id_of_proofs[used_rules_string].append(i)

        with open(self.get_dataset_path(self.input_file), 'r', encoding='utf-8') as infile:
            infile_lines = infile.readlines()
            with open(self.get_dataset_path(output_file), 'w', encoding='utf-8') as outfile:
                for key, datas in id_of_proofs.items():
                    selected = random.sample(datas, min(len(datas), sample_num))
                    rows_to_be_kept.extend(selected)
                    for index in selected:
                        outfile.write(infile_lines[index])

        time_taken = time.time() - start_time
        self.record_deduplicate_information(output_file, total_rows, rows_to_be_kept, time_taken)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_file = 'geometry_clauses15_samples10M_aux.jsonl'
    else:
        input_file = sys.argv[1]

    if len(sys.argv) < 3:
        sample_num = 10
    else:
        sample_num = int(sys.argv[2])

    analyzer = GeometryEquivalenceAnalyzer(input_file)
    analyzer.deduplicate_by_rules(sample_num)