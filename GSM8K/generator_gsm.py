import os
import shutil
import sys

import yaml

from templates_gsm import (
    COMBINE_SCRIPT_TEMPLATE,
    EVAL_SCRIPT_TEMPLATE,
    FILTER_SCRIPT_TEMPLATE,
    GENERATION_SCRIPT_TEMPLATE,
    PROCESS_SCRIPT_TEMPLATE,
    TRAIN_TEMPLATE,
)


def generate_scripts(task_filename):
    # Generate train script code
    with open(task_filename, "r") as f:
        initial_config = yaml.safe_load(f)
    os.makedirs((initial_config["base_path"]), exist_ok=True)

    # Set base to be the current working directory
    base = os.getcwd()
    initial_config["base"] = base

    # Copy the seed file to step 0
    step_0_data_directory = f'{initial_config["base_path"]}/data/step_0/'
    os.makedirs((step_0_data_directory), exist_ok=True)
    shutil.copy(
        initial_config["seed_data_path"], step_0_data_directory + "step_0.json"
    )

    scripts_to_run = []
    scripts_directory = f'{initial_config["base_path"]}/scripts'
    os.makedirs((scripts_directory), exist_ok=True)

    total_steps = initial_config["steps"]
    for i in range(total_steps):
        data_output_directory = f'{initial_config["base_path"]}/data/step_{i}/'
        data_next_output_directory = f'{initial_config["base_path"]}/data/step_{i+1}/'

        data_path = data_output_directory + f"step_{i}.json"
        model_output_path = f'{initial_config["base_path"]}/model/step_{i}/'

        os.makedirs((os.path.dirname(data_path)), exist_ok=True)
        os.makedirs((data_next_output_directory), exist_ok=True)
        os.makedirs((data_output_directory), exist_ok=True)
        os.makedirs((model_output_path), exist_ok=True)

        config = initial_config.copy()
        config["data_path"] = data_path
        config["output_path"] = model_output_path

        # need to change the data and output paths
        train_script = TRAIN_TEMPLATE.format_map(config)
        train_script_path = f"{scripts_directory}/train_step_{i}.sh"
        scripts_to_run.append(train_script_path)
        with open(train_script_path, "w") as f:
            f.write(train_script)

        eval_output_directory = f'{initial_config["base_path"]}/eval/step_{i}/'
        os.makedirs((eval_output_directory), exist_ok=True)

        # Generate evaluation scripts to evaluate on original train and test set
        eval_script_path = f"{scripts_directory}/eval_step_{i}.sh"
        scripts_to_run.append(eval_script_path)
        with open(eval_script_path, "w") as f:
            f.write(
                EVAL_SCRIPT_TEMPLATE.format(
                    test_mode="train", # ignored
                    extra_flags=f"--file {data_path}",
                    model_path=model_output_path,
                    eval_output_directory=eval_output_directory,
                    **initial_config,
                )
            )
            f.write("\n")

        test_script_path = f"{scripts_directory}/test_step_{i}.sh"
        scripts_to_run.append(test_script_path)
        with open(test_script_path, "w") as f:
            f.write(
                EVAL_SCRIPT_TEMPLATE.format(
                    test_mode="test",
                    extra_flags="",
                    model_path=model_output_path,
                    eval_output_directory=eval_output_directory,
                    **initial_config,
                )
            )

        # we are done as there is no need to run the generation script on the last step
        if i == total_steps - 1:
            break

        # Filter out wrong answers to pass to generation script
        filtered_data_path = data_output_directory + "filtered.json"
        filter_script_path = f"{scripts_directory}/filter_step_{i}.sh"
        scripts_to_run.append(filter_script_path)
        with open(filter_script_path, "w") as f:
            f.write(FILTER_SCRIPT_TEMPLATE.format(input_data_path=eval_output_directory + "file_0.jsonl", output_data_path=filtered_data_path, **initial_config))

        # Run stanford alpaca code on the wrong answers
        # generates a regen.json in this output directory
        data_generation_script_path = f"{scripts_directory}/data_generation_step_{i}.sh"
        scripts_to_run.append(data_generation_script_path)
        with open(data_generation_script_path, "w") as f:
            f.write(
                GENERATION_SCRIPT_TEMPLATE.format_map(
                    {
                        "seed_path": filtered_data_path,
                        "generation_output_directory": data_output_directory,
                        **initial_config,
                    }
                )
            )

        # Combine this with the old data
        process_output_path = data_output_directory + "/regen_processed.json"
        process_output_script_path = f"{scripts_directory}/process_combine_step_{i}.sh"
        scripts_to_run.append(process_output_script_path)
        with open(process_output_script_path, "w") as f:
            f.write(
                PROCESS_SCRIPT_TEMPLATE.format(
                    process_output_path=process_output_path,
                    process_input_path=data_output_directory + "/regen.json",
                    **initial_config,
                )
            )
            f.write("\n")
            f.write(
                COMBINE_SCRIPT_TEMPLATE.format(
                    combine_output_path=data_next_output_directory
                    + f"/step_{i+1}.json",
                    combine_input_path=data_path,  # data from this step
                    combine_input_path_2=process_output_path,
                    **initial_config,
                )
            )

    # Write a bash script called run_all.sh that runs all the scripts, and when each script finishes logs the script's name to script_log.txt
    # Each script should not be run if it's already in script_log.txt
    # If one script fails, abort
    run_all_script_path = f'{initial_config["base_path"]}/run_all.sh'
    with open(run_all_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("touch script_log.txt\n")
        for script in scripts_to_run:
            f.write(f"if ! grep -q {script} script_log.txt; then\n")
            f.write(f"  echo Running {script}\n")
            f.write(f"  {script}\n")
            f.write(f"  echo {script} >> script_log.txt\n")
            f.write("fi\n")
    os.chmod(run_all_script_path, 0o755)

    # write the script order to a file
    script_order_path = f'{initial_config["base_path"]}/script_order.txt'
    with open(script_order_path, "w") as f:
        for script in scripts_to_run:
            f.write(f"{script}\n")

    # Iterate over files in directory
    for filename in os.listdir(scripts_directory):
        if filename.endswith(".sh"):
            filepath = os.path.join(scripts_directory, filename)

            # Change permission to make the file executable
            os.chmod(filepath, 0o755)


if __name__ == "__main__":
    generate_scripts(sys.argv[1])
