# Chapter 1: Configuration

Welcome to the `crest` project! This tutorial will guide you through the core concepts that make up a `crest` workflow. We'll start with something fundamental: managing your project's settings.

Imagine you're building something complex, like a robot or even just baking a complicated cake. You need instructions! Where do you find the ingredients? How long do you bake? What temperature?

In the world of software, especially for projects that involve processing data, training models, or running simulations, you also need instructions. Where is the data located? Which model should we use? How many items should we process at once? These settings change depending on what you're doing, and you need a good way to organize and access them.

This is where the **Configuration** abstraction comes in.

## What is Configuration in CREST?

Think of the `Config` component in `crest` as your project's main instruction book or **recipe book**. It's designed to hold all the important settings and parameters your project needs to run.

Instead of hardcoding these settings directly in your code (which makes it hard to change things later), you store them in a separate file. `crest` uses YAML files for this, which are easy for both humans and computers to read.

The `Config` component helps you:

1.  **Load:** Read settings from a YAML file when your project starts.
2.  **Store:** Keep all settings in one place within your running program.
3.  **Access:** Easily get any setting whenever you need it.
4.  **Update:** Change settings while the program is running if necessary.
5.  **Save:** Write the current settings back to a file.

This single source of truth ensures that all parts of your `crest` project are using the same "recipe."

## Your First Recipe: A Simple Use Case

Let's imagine a very simple `crest` workflow. We need to process some data that lives in a specific folder. Our recipe needs to tell us:

1.  Where to find the raw data.
2.  Where to save the processed data.

Let's create a simple configuration file named `my_project_config.yaml`:

```yaml
# my_project_config.yaml
data_paths:
  raw_data_directory: /path/to/your/raw/data
  processed_data_directory: /path/to/your/processed/data

processing_settings:
  batch_size: 32
  enable_logging: True
```

This is our simple recipe! It's a YAML file with two main sections: `data_paths` and `processing_settings`, each containing specific values.

## Using the Config Component

Now, how do we use this recipe file in our `crest` project? We use the `Config` class.

First, make sure you have a YAML file like the one above. Replace `/path/to/your/raw/data` and `/path/to/your/processed/data` with actual paths if you want to run this, or just keep them as is for the example.

Here's how you would load this configuration using `crest.Config`:

```python
# Example: Using the Config
from crest import Config

# Assume 'my_project_config.yaml' exists in the same directory
config_file_path = 'my_project_config.yaml'

# Create a Config object, telling it which file to load
project_config = Config(config_file_path)

print("Configuration loaded successfully!")
# The configuration is now stored inside 'project_config'
```

**Explanation:**

*   We import the `Config` class from the `crest` library.
*   We specify the path to our YAML file.
*   We create an instance of the `Config` class, passing the file path to its `__init__` method.
*   The `Config` object automatically loads the settings from the file when it's created.

Now that the configuration is loaded, how do we get the values we need? For example, how do we find the `raw_data_directory`? The `Config` object allows you to access settings using dot notation, similar to accessing attributes of an object, or like a dictionary.

```python
# Example: Accessing settings
# Assuming project_config was created as in the previous snippet

# Accessing settings using dot notation (very convenient!)
raw_dir = project_config.data_paths.raw_data_directory
batch_size = project_config.processing_settings.batch_size

print(f"Raw data directory: {raw_dir}")
print(f"Batch size: {batch_size}")

# You can also access like a dictionary, though dot notation is often preferred
# raw_dir_dict_access = project_config['data_paths']['raw_data_directory']
# print(f"Raw data directory (dict access): {raw_dir_dict_access}")
```

**Explanation:**

*   `project_config.data_paths` gives you access to the dictionary stored under the `data_paths` key in your YAML.
*   `.raw_data_directory` then accesses the value associated with `raw_data_directory` within that dictionary.
*   This makes reading your settings very clean and readable!

What if you need to change a setting during your program's execution? The `Config` object has an `update` method, or you can modify the internal dictionary directly (though `update` is often cleaner for merging).

```python
# Example: Updating settings
# Assuming project_config is loaded

print(f"Original batch size: {project_config.processing_settings.batch_size}")

# Update settings using the update method
project_config.update({
    'processing_settings': {
        'batch_size': 64 # Let's change the batch size
    }
})

# Or directly modify the dictionary (use with care)
# project_config._config['processing_settings']['batch_size'] = 128


print(f"Updated batch size: {project_config.processing_settings.batch_size}")
```

**Explanation:**

*   The `update` method takes a dictionary and merges it into the current configuration dictionary. Be careful with nested structures; you need to provide the full path in the update dictionary if you want to replace or add nested values.
*   After updating, accessing the same key (`project_config.processing_settings.batch_size`) gives you the new value.

Finally, if you make changes and want to save them back to a file (maybe for future runs), you can use the `save` method:

```python
# Example: Saving settings
# Assuming project_config has potentially been updated

# Save back to the original file
project_config.save()

# Or save to a new file
# project_config.save('updated_config.yaml')

print(f"Configuration saved.")
```

**Explanation:**

*   Calling `save()` without arguments writes the current state of `project_config` back to the file it was originally loaded from (`my_project_config.yaml` in our example).
*   You can provide a `filepath` to save it to a different location.

## Behind the Scenes: How Config Works

How does this magic happen? Let's peek under the hood of the `crest.Config` class.

The core idea is simple: the `Config` object wraps around a standard Python dictionary (`self._config`). When you load a YAML file, the contents are parsed and stored in this dictionary. When you access settings, the `Config` object looks up the values in this dictionary.

Here's a simplified step-by-step flow when you create a `Config` object with a file and then access a setting:

```{mermaid}
sequenceDiagram
    participant User as You (Code)
    participant ConfigObj as Config Object
    participant YAMLFile as my_project_config.yaml

    User->>ConfigObj: Create Config(config_file_path)
    ConfigObj->>ConfigObj: Initialize empty dictionary (_config)
    ConfigObj->>ConfigObj: Check if config_file_path exists
    alt File exists
        ConfigObj->>ConfigObj: Call load()
        ConfigObj->>YAMLFile: Read file contents
        YAMLFile-->>ConfigObj: Return YAML text
        ConfigObj->>ConfigObj: Parse YAML text (using PyYAML)
        ConfigObj->>ConfigObj: Store parsed data in _config dictionary
    end
    ConfigObj-->>User: Config object ready

    User->>ConfigObj: Access a setting (e.g., project_config.data_paths.raw_data_directory)
    ConfigObj->>ConfigObj: Check if 'data_paths' is in _config (using __getattr__)
    ConfigObj->>ConfigObj: Return the dictionary for 'data_paths'
    User->>ConfigObj: Access nested setting (.raw_data_directory)
    ConfigObj->>ConfigObj: (Now looking at the returned dictionary) Check if 'raw_data_directory' is in this dictionary (standard dict access)
    ConfigObj->>ConfigObj: Return the value associated with 'raw_data_directory'
    ConfigObj-->>User: Provide the setting value
```

Let's look at the relevant parts of the code in `configuration/Config.py`.

The `__init__` method sets up the object and handles loading:

```python
# From configuration/Config.py (simplified)
class Config:
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self._config = {} # This dictionary holds your settings

        if (config_file and os.path.exists(config_file)):
            # If a file is given and exists, load it
            self._config = self.load()
```

This simply initializes the internal `_config` dictionary and calls the `load` method if a file was provided.

The `load` method does the actual file reading and parsing:

```python
# From configuration/Config.py (simplified)
import yaml # Need this to work with YAML

class Config:
    # ... __init__ method ...

    def load(self, config_file: str = None):
        # ... handle file path ...

        with open(self.config_file, 'r') as file:
            # yaml.safe_load reads the YAML file and turns it into Python dictionaries/lists
            self._config = yaml.safe_load(file)

        return self._config
    # ... other methods ...
```

This snippet shows the core loading logic: open the file and use `yaml.safe_load` to convert the YAML text into Python data structures (mostly dictionaries).

The neat dot-notation access (`project_config.some_setting`) is handled by a special Python method called `__getattr__`:

```python
# From configuration/Config.py (simplified)
class Config:
    # ... __init__, load, save, update methods ...

    def __getattr__(self, name):
        """ Forwards attribute access to the underlying config dict. """

        if name in self._config:
            # If the requested name (like 'data_paths') is a key in the dictionary, return its value
            return self._config[name]

        # If the name is not a key, raise an error like a normal object would
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    # ... other methods ...
```

When you try to access `project_config.some_setting`, Python first looks for a normal attribute named `some_setting` on the `Config` object. If it doesn't find one, it calls `__getattr__('some_setting')`. This method then checks if `'some_setting'` exists as a key in the `_config` dictionary and returns the corresponding value if it does. This is how you get the convenient dot-notation access for your settings!

## Conclusion

In this chapter, you learned that the `Configuration` component in `crest` is like your project's recipe book, storing essential settings in a YAML file. You saw how to:

*   Create a simple configuration file.
*   Load the configuration using the `Config` class.
*   Access settings easily using dot notation.
*   Update and save settings.

Understanding how to manage configuration is a crucial first step in building any structured workflow. Now that you know how to tell your project where to find things and how to behave, the next step is to understand how `crest` handles the data itself.

Ready to learn about data? Let's move on to the next chapter: [Data Abstractions](02_data_abstractions_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)