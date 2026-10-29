import importlib


def env_settings():
    module = importlib.import_module("lib.train.admin.local")
    return module.EnvironmentSettings()
