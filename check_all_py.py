import os
import ast
import sys

def check_python_files(repo_path):
    total_files = 0
    passed_files = 0
    failed_files = []

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source)
                    passed_files += 1
                except SyntaxError as e:
                    failed_files.append((filepath, str(e)))
                except Exception as e:
                    failed_files.append((filepath, str(e)))

    print(f"Total Python files checked: {total_files}")
    print(f"Passed: {passed_files}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("Failures:")
        for file, err in failed_files:
            print(f"- {file}: {err}")
        sys.exit(1)
    else:
        print("All Python files are syntactically valid.")

if __name__ == "__main__":
    check_python_files(r"d:\github\Rakshak")
