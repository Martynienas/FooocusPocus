import os
import unittest
from unittest import mock

import torch

from modules import zimage_poc


class _DummyTransformer:
    def __init__(self):
        self.dtype = torch.float32
        self.config = type("cfg", (), {"in_channels": 4})()


class _DummyPipelineWithLatents:
    def __init__(self):
        self.transformer = _DummyTransformer()
        self.vae_scale_factor = 8

    def __call__(self, *, latents=None, **kwargs):
        return latents, kwargs


class _DummyPipelineWithoutLatents:
    def __init__(self):
        self.transformer = _DummyTransformer()
        self.vae_scale_factor = 8

    def __call__(self, *, prompt=None, **kwargs):
        return prompt, kwargs


class TestZImageAltPath(unittest.TestCase):
    def _base_kwargs(self):
        return dict(
            source_kind="single_file",
            source_path="/tmp/dummy.safetensors",
            flavor="turbo",
            checkpoint_folders=[],
            prompt="p",
            negative_prompt="",
            width=832,
            height=1216,
            steps=9,
            guidance_scale=1.0,
            seed=1234,
        )

    def test_dispatch_routes_to_alternate_when_enabled(self):
        kwargs = self._base_kwargs()
        with mock.patch.dict(os.environ, {"FOOOCUS_ZIMAGE_ALT_PATH": "1"}, clear=False):
            with mock.patch("modules.zimage_poc._generate_zimage_alt", return_value="alt") as alt_mock:
                with mock.patch("modules.zimage_poc._generate_zimage_legacy", return_value="legacy") as legacy_mock:
                    result = zimage_poc.generate_zimage(**kwargs)
        self.assertEqual("alt", result)
        alt_mock.assert_called_once()
        legacy_mock.assert_not_called()

    def test_dispatch_routes_to_legacy_when_disabled(self):
        kwargs = self._base_kwargs()
        with mock.patch.dict(os.environ, {"FOOOCUS_ZIMAGE_ALT_PATH": "0"}, clear=False):
            with mock.patch("modules.zimage_poc._generate_zimage_alt", return_value="alt") as alt_mock:
                with mock.patch("modules.zimage_poc._generate_zimage_legacy", return_value="legacy") as legacy_mock:
                    result = zimage_poc.generate_zimage(**kwargs)
        self.assertEqual("legacy", result)
        legacy_mock.assert_called_once()
        alt_mock.assert_not_called()

    def test_dispatch_defaults_to_alternate_when_env_unset(self):
        kwargs = self._base_kwargs()
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("modules.zimage_poc._generate_zimage_alt", return_value="alt") as alt_mock:
                with mock.patch("modules.zimage_poc._generate_zimage_legacy", return_value="legacy") as legacy_mock:
                    result = zimage_poc.generate_zimage(**kwargs)
        self.assertEqual("alt", result)
        alt_mock.assert_called_once()
        legacy_mock.assert_not_called()

    def test_alt_force_full_gpu_default_is_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(zimage_poc._zimage_alt_force_full_gpu_enabled())
        with mock.patch.dict(os.environ, {"FOOOCUS_ZIMAGE_ALT_FORCE_FULL_GPU": "0"}, clear=False):
            self.assertFalse(zimage_poc._zimage_alt_force_full_gpu_enabled())

    def test_alt_latent_source_defaults_to_gpu(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual("gpu", zimage_poc._zimage_alt_latent_source_mode())
        with mock.patch.dict(os.environ, {"FOOOCUS_ZIMAGE_ALT_LATENT_SOURCE": "cpu"}, clear=False):
            self.assertEqual("cpu", zimage_poc._zimage_alt_latent_source_mode())

    def test_alt_path_rejects_pipeline_without_latents_kwarg(self):
        pipeline = _DummyPipelineWithoutLatents()
        with self.assertRaisesRegex(RuntimeError, "latents support"):
            zimage_poc._ensure_alt_path_prerequisites(pipeline, width=832, height=1216)

    def test_latents_are_deterministic_for_same_seeds(self):
        pipeline = _DummyPipelineWithLatents()
        with mock.patch.dict(os.environ, {"FOOOCUS_ZIMAGE_ALT_LATENT_SOURCE": "cpu"}, clear=False):
            latents_a = zimage_poc._build_latents_from_seeds(
                pipeline=pipeline,
                seed_list=[101, 202],
                width=832,
                height=1216,
                generator_device="cpu",
            )
            latents_b = zimage_poc._build_latents_from_seeds(
                pipeline=pipeline,
                seed_list=[101, 202],
                width=832,
                height=1216,
                generator_device="cpu",
            )
            latents_c = zimage_poc._build_latents_from_seeds(
                pipeline=pipeline,
                seed_list=[303, 202],
                width=832,
                height=1216,
                generator_device="cpu",
            )
        self.assertTrue(torch.equal(latents_a, latents_b))
        self.assertFalse(torch.equal(latents_a, latents_c))

    def test_alt_path_random_source_uses_latents_not_generator(self):
        pipeline = _DummyPipelineWithLatents()
        call_kwargs = {
            "width": 832,
            "height": 1216,
            "generator": object(),
        }
        zimage_poc._set_generation_random_source(
            call_kwargs=call_kwargs,
            seed_list=[7, 11],
            pipeline=pipeline,
            generator_device="cpu",
            use_alt_path=True,
        )
        self.assertNotIn("generator", call_kwargs)
        self.assertIn("latents", call_kwargs)
        self.assertEqual(2, int(call_kwargs["latents"].shape[0]))


if __name__ == "__main__":
    unittest.main()
