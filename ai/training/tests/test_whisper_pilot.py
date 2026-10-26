from __future__ import annotations
import importlib.util
import unittest

from pbl4_training.train_whisper_pilot import prepare_labels, strict_digits, score_training_predictions
from pbl4_training.dataset import DatasetError


class WhisperPilotTests(unittest.TestCase):
    def test_labels_preserve_eos_without_duplicate_decoder_start(self):
        self.assertEqual(prepare_labels([1, 3, 4, 2], 1), [3, 4, 2])
        self.assertEqual(prepare_labels([3, 4, 2], 1), [3, 4, 2])

    def test_strict_digits_preserves_zero_and_rejects_extra_words(self):
        self.assertEqual(strict_digits("không một hai ba bốn năm."), "012345")
        self.assertEqual(strict_digits("0, 1, 2, 3, 4, 5"), "012345")
        self.assertIsNone(strict_digits("mã là 012345"))
        self.assertIsNone(strict_digits("12345"))

    def test_reused_metrics_are_not_labeled_held_out_accuracy(self):
        rows = [{"transcript": "không một hai ba bốn năm", "otp_digits": "012345"}]
        metrics = score_training_predictions(rows, [{"transcript": "012345"}])
        self.assertEqual(metrics["exact_otp_count"], 1)
        self.assertFalse(metrics["held_out_evaluation"])
        self.assertEqual(metrics["split"], "train_reuse")
        with self.assertRaises(DatasetError):
            score_training_predictions(rows, [])

    @unittest.skipIf(importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None, "Requires Whisper training dependencies")
    def test_actual_whisper_decoder_gradients_change_weights_but_not_encoder(self):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from transformers.modeling_outputs import BaseModelOutput
        torch.set_num_threads(2)
        config = WhisperConfig(vocab_size=64, num_mel_bins=80, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
            max_source_positions=16, max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            decoder_start_token_id=1)
        model = WhisperForConditionalGeneration(config)
        for p in model.model.encoder.parameters():
            p.requires_grad_(False)
        before_encoder = {k: v.clone() for k, v in model.model.encoder.state_dict().items()}
        before_decoder = {k: v.clone() for k, v in model.model.decoder.state_dict().items()}
        with torch.no_grad():
            encoded = model.model.encoder(torch.randn(1, 80, 32)).last_hidden_state
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded), labels=torch.tensor([[3, 4, 2]]), use_cache=False).loss
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.equal(v, before_encoder[k]) for k, v in model.model.encoder.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, before_decoder[k]) for k, v in model.model.decoder.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
