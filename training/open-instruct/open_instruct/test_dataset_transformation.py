import json
import unittest
from unittest import mock

import open_instruct.dataset_transformation


class TestEnvConfigNormalization(unittest.TestCase):
    def test_normalize_single_dict_env_config(self):
        row = {"env_config": {"env_name": "guess_number", "number": "7"}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "guess_number", "number": "7"}]})

    def test_normalize_list_env_config(self):
        row = {"env_config": [{"env_name": "counter", "target": "3"}]}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "counter", "target": "3"}]})

    def test_normalize_canonical_env_config(self):
        row = {"env_config": {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(
            row["env_config"], {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}
        )


class TestConfigHash(unittest.TestCase):
    def test_load_dataset_configs_passes_hf_config_name(self):
        dataset = open_instruct.dataset_transformation.Dataset.from_dict({"messages": [[]]})
        with (
            mock.patch("open_instruct.dataset_transformation.get_commit_hash", return_value="abc123"),
            mock.patch("open_instruct.dataset_transformation.load_dataset", return_value=dataset) as load_dataset,
        ):
            dcs = open_instruct.dataset_transformation.load_dataset_configs(
                dataset_mixer_list=["org/repo", "1.0"],
                dataset_mixer_list_splits=["train"],
                dataset_transform_fn=[],
                transform_fn_args=[],
                dataset_mixer_list_config_names=["named_config"],
            )

        self.assertEqual(dcs[0].dataset_config_name, "named_config")
        load_dataset.assert_called_once()
        args, kwargs = load_dataset.call_args
        self.assertEqual(args[:2], ("org/repo", "named_config"))
        self.assertEqual(kwargs["split"], "train")


class TestToolNormalization(unittest.TestCase):
    def test_normalize_tools_accepts_json_encoded_schema_list(self):
        tool_schema = {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        tools = open_instruct.dataset_transformation._normalize_tools_for_chat_template(json.dumps([tool_schema]))

        self.assertEqual(tools, [tool_schema])

    def test_normalize_tools_rejects_tool_name_lists(self):
        with self.assertRaises(TypeError):
            open_instruct.dataset_transformation._normalize_tools_for_chat_template(["bash"])


if __name__ == "__main__":
    unittest.main()
