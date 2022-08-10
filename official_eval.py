import os

# EVAL_DIR = "eval"


def official_f1(EVAL_DIR, eval_script):

    # Run the perl script
    try:
        cmd = "perl {1} {0}/proposed_answers.txt {0}/answer_keys.txt > {0}/result.txt".format(
            EVAL_DIR , eval_script
        ) #TODO copy script to the folder
        print(cmd)
        os.system(cmd)
    except:
        raise Exception("perl is not installed or proposed_answers.txt is missing")

    with open(os.path.join(EVAL_DIR, "result.txt"), "r", encoding="utf-8") as f:
        try:
            macro_result = list(f)[-1]
            macro_result = macro_result.split(":")[1].replace(">>>","").strip()  # TODO result.txt is empty --> UnboundLocalError: local variable 'macro_result' referenced before assignment
            macro_result = macro_result.split("=")[1].strip().replace("%", "")
            macro_result = float(macro_result) / 100
        except IndexError as e:
            print(e)

    return macro_result


if __name__ == "__main__":
    print("macro-averaged F1 = {}%".format(official_f1() * 100))
