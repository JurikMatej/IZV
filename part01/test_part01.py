#!/usr/bin/env python3
"""
Skript pro automaticke testovani prvni casti projektu.

Spousteni:
   pytest
nebo
   python3 -m pytest
"""
import os
import pytest
import part01


def test_integrate():
    """Test vypoctu integralu """

    def f(x):
        return 10 * x + 2

    r = part01.integrate(f, 0, 1, 100)
    assert r == pytest.approx(7)


def test_generate_fn():
    """Test generovani grafu s vice funkcemi"""
    part01.generate_graph([1., 1.5, 2.], show_figure=False,
                          save_path="tmp_fn.png")
    assert os.path.exists("tmp_fn.png")


def test_generate_sin():
    """Test generovani grafu se sinusovkami"""
    part01.generate_sinus(show_figure=False, save_path="tmp_sin.png")
    assert os.path.exists("tmp_sin.png")


def test_download():
    """Test stazeni dat"""
    data = part01.download_data()

    assert len(data) == 40
    assert data[0]["position"] == "Cheb"
    assert data[0]["lat"] == pytest.approx(50.0683)
    assert data[0]["long"] == pytest.approx(12.3913)
    assert data[0]["height"] == pytest.approx(483.0)
