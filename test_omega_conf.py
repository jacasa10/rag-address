from omegaconf import OmegaConf

conf = OmegaConf.load("test_yaml.yml")

print(conf["hello"])
