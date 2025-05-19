import os

# Access all environment variables using os.environ
environment_variables = os.environ

# Iterate through the environment variables and print them
for key, value in environment_variables.items():
    print(f"{key}={value}")