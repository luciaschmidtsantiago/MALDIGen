import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Alternatively, can be read from csv files or from bruker output data.
    Reading from Bruker data is based on the code in https://github.com/sgibb/readBrukerFlexData

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        """
        Reads a spectrum from a text file and removes any **non-numeric** headers.

        Parameters:
        - file (str): Path to the spectrum file.
        - sep (str, optional): Separator used in the file (default is " ").

        Returns:
        - SpectrumObject instance.
        """
        # Read the TSV file
        s = pd.read_table(file, sep=sep, index_col=None, comment="#", header=None).values

        # Check the first row and remove if it's not numeric
        if not str(s[0, 0]).replace(".", "", 1).isdigit():
            s = s[1:]  # Remove the first row

        # Convert to float to ensure no string issues
        mz = s[:, 0].astype(float)
        intensity = s[:, 1].astype(float)

        return cls(mz=mz, intensity=intensity)
