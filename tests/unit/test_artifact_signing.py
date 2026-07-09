"""S21A — Tests for artifact/manifest signing (ArtifactSigner)."""

from __future__ import annotations

import pytest

from trainer.research_lab.signing import (
    ArtifactSigner,
    SigningKeyError,
)


@pytest.fixture()
def keypair() -> tuple[str, str]:
    return ArtifactSigner.generate_keypair()


class TestKeyAvailability:
    def test_no_keys_configured(self) -> None:
        signer = ArtifactSigner(private_key_pem="", public_key_pem="")
        assert signer.can_sign is False
        assert signer.can_verify is False

    def test_keys_from_explicit_args(self, keypair) -> None:
        priv, pub = keypair
        signer = ArtifactSigner(private_key_pem=priv, public_key_pem=pub)
        assert signer.can_sign is True
        assert signer.can_verify is True

    def test_keys_from_env(self, monkeypatch, keypair) -> None:
        priv, pub = keypair
        monkeypatch.setenv("ARTIFACT_SIGNING_PRIVATE_KEY", priv)
        monkeypatch.setenv("ARTIFACT_SIGNING_PUBLIC_KEY", pub)
        signer = ArtifactSigner()
        assert signer.can_sign is True
        assert signer.can_verify is True


class TestSignAndVerify:
    def test_sign_then_verify_succeeds(self, keypair) -> None:
        priv, pub = keypair
        signer = ArtifactSigner(private_key_pem=priv, public_key_pem=pub)
        data = b"artifact-bytes-12345"
        signature = signer.sign(data)
        assert signer.verify(data, signature) is True

    def test_verify_rejects_tampered_data(self, keypair) -> None:
        priv, pub = keypair
        signer = ArtifactSigner(private_key_pem=priv, public_key_pem=pub)
        signature = signer.sign(b"original-bytes")
        assert signer.verify(b"tampered-bytes", signature) is False

    def test_verify_rejects_signature_from_different_key(self, keypair) -> None:
        priv, pub = keypair
        other_priv, _ = ArtifactSigner.generate_keypair()
        signer_a = ArtifactSigner(private_key_pem=other_priv)
        signer_b = ArtifactSigner(private_key_pem=priv, public_key_pem=pub)
        data = b"some-data"
        signature = signer_a.sign(data)
        assert signer_b.verify(data, signature) is False

    def test_verify_rejects_malformed_hex(self, keypair) -> None:
        _, pub = keypair
        signer = ArtifactSigner(public_key_pem=pub)
        assert signer.verify(b"data", "not-hex!!") is False

    def test_verify_rejects_empty_signature(self, keypair) -> None:
        _, pub = keypair
        signer = ArtifactSigner(public_key_pem=pub)
        assert signer.verify(b"data", "") is False

    def test_sign_without_private_key_raises(self) -> None:
        signer = ArtifactSigner(private_key_pem="")
        with pytest.raises(SigningKeyError):
            signer.sign(b"data")

    def test_verify_without_public_key_raises(self) -> None:
        signer = ArtifactSigner(public_key_pem="")
        with pytest.raises(SigningKeyError):
            signer.verify(b"data", "aabbcc")

    def test_malformed_private_key_raises(self) -> None:
        signer = ArtifactSigner(private_key_pem="not a real PEM key")
        with pytest.raises(SigningKeyError):
            signer.sign(b"data")

    def test_malformed_public_key_raises(self) -> None:
        signer = ArtifactSigner(public_key_pem="not a real PEM key")
        with pytest.raises(SigningKeyError):
            signer.verify(b"data", "aabbcc")
