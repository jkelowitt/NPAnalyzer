import pandas as pd
import numpy as np
import glob
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy.interpolate as si
import scipy.signal as ss
import scipy.optimize as so
import matplotlib.pyplot as plt


def get_data(filename):
    # Make dataframe
    df = pd.read_csv(filename, names=["Wavelength", "Absorbance"], delimiter=",")[::-1]

    # Remove rows where we get NaN as an absorbance
    df = df.apply(pd.to_numeric, errors='raise')
    df = df.dropna()

    # We only care about the wavelengths between 400 and 700 nm.
    df = df[(df["Wavelength"] < 700) & (400 < df["Wavelength"])]

    # Smooth the absorbance data
    df["Absorbance"] = ss.savgol_filter(df["Absorbance"], window_length=9, polyorder=4)

    # An object which makes splines
    spline_maker = IUS(df["Wavelength"], df["Absorbance"])

    # An object which calculates the derivative of those splines
    deriv_1 = spline_maker.derivative(n=1)
    deriv_2 = spline_maker.derivative(n=2)

    # Calculate the derivative at all of the splines
    df["First_derivative"] = deriv_1(df["Wavelength"])
    df["Second_derivative"] = deriv_2(df["Wavelength"])

    return df


"""Calculation functions"""


def calc_offset(m, x1, y1):
    return x1 - (y1 / m)


def find_offset_wavelength(df, filename):
    goal = np.min(df["First_derivative"])  # Find the value of the largest downward slope

    inflection_point = float(df[df["First_derivative"] == goal]["Wavelength"])

    # Find height of inflection point
    inflection_height = float(df[df["First_derivative"] == goal]["Absorbance"])

    # Get the value of the offset wavelength
    offset = calc_offset(goal, inflection_point, inflection_height)

    return offset


def light_energy(wavelength, h=6.6262e-34, c=299792458):
    """
    wavelength --> Wavelength of light in nm
    h --> Plancks Constant
    c --> Speed of light in meters/second
    """
    return (h * c) / (wavelength * 10E-9)


def quadratic(A, B, C):
    return (-B - np.sqrt(B ** 2 - (4 * A * C))) / (2 * A)


def find_radius(E, me, mh, ee=-1.602e-19, h=6.626e-34, eps=10.6, eps0=8.857e-12):
    """
    E --> Band Gap energy
    ee --> Charge of an electron
    me --> Ratio of the mass an excited electron to an electron
    mh --> Ratio of the mass an excited hole to an electron
    h --> Plancks Constant
    eps --> Dielectric constant of CdSe
    eps0 --> Vacuum Permitivity
    """
    mass_electron = 9.11E-31  # kg
    me *= mass_electron  # kg
    mh *= mass_electron  # kg
    A = ((h ** 2) / 8) * ((1 / me) + (1 / mh))
    B = (-1.8 * (ee ** 2)) / (4 * np.pi * eps * eps0)
    C = -E
    return quadratic(C, B, A)


"""Gaussian Functions"""


def inf_peak_pairs(xdata, ydata, order=11):
    # Get derivatives
    deriv1 = ss.savgol_filter(ydata, window_length=order, polyorder=2, deriv=1)
    deriv2 = ss.savgol_filter(ydata, window_length=order, polyorder=2, deriv=2)

    # Create interpolation objects
    interdata = si.interp1d(xdata, ydata, kind=2)
    interderiv1 = si.interp1d(xdata, deriv1, kind=2)
    interderiv2 = si.interp1d(xdata, deriv2, kind=2)

    # Find roots, peaks, and inflection points
    roots = np.array(find_roots(xdata, interderiv1))
    peaks = np.array(find_peaks(interdata, interderiv2, roots))
    inflections = np.array(find_roots(xdata, interderiv2))

    # Find the inflection points closest to each peak
    inf_peak_pairs = []  # (peak, inflection point below, inflection point above)
    for item in peaks:
        try:
            above = inflections[inflections > item].min()  # Smallest number above
            below = inflections[inflections < item].max()  # Largest number below
            inf_peak_pairs.append((item, below, above))
        except:
            print("Peaks not found")

    return inf_peak_pairs


def gaussian_sum(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        center = params[i]
        amplitude = params[i + 1]
        width = params[i + 2]

        this_gaussian = amplitude * np.exp(-((x - center) / width) ** 2)

        y = y + this_gaussian
    return y


def gaussian_approximations(infs_peaks, xdata, ydata, interdata, filename):
    guess = []  # (Peak positions, amplitude, width)
    for peak, below, above in infs_peaks:
        if 400 <= peak <= 600:
            guess.append(peak)  # Peak location
            guess.append(interdata(peak))  # Amplitude
            guess.append(above - below)  # Width
        else:  # Ignore minor peaks outside the given range.
            pass

    if not guess:  # What if the local maximum is one of the end points?
        # Get 2d derivative and Create interpolation object
        deriv2 = ss.savgol_filter(ydata, window_length=11, polyorder=2, deriv=2)
        interderiv2 = si.interp1d(xdata, deriv2, kind=2)

        peak = min(xdata)  # Assuming the far left is the local maximum
        amplitude = max(ydata)
        inflection = np.array(find_roots(xdata, interderiv2))[0]  # The inflection point we care about is the first one.
        width = 2 * (inflection - peak)

        guess.append(peak)
        guess.append(amplitude)
        guess.append(width)

    opt_guess, opt_conv = so.curve_fit(gaussian_sum,
                                       xdata,
                                       ydata,
                                       p0=guess,
                                       maxfev=1_000_000,
                                       bounds=(0, 1000))

    print("Accepted Optimizations (x, a, w): ", opt_guess, "\n")
    return opt_guess


def show_gaussians(x, data, params):
    plt.plot(x, data)
    if params is not None:  # If there are parameters, plot them.
        for i in range(0, len(params), 3):
            center = params[i]
            amplitude = params[i + 1]
            width = params[i + 2]

            this_gaussian = amplitude * np.exp(-((x - center) / width) ** 2)
            plt.plot(x, this_gaussian, dashes=[6, 2])

            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Absorbance")


def find_roots(wavelength, interderiv):
    roots = []

    for value in wavelength:
        try:
            new_root = so.brentq(interderiv, value, value + 1)
            roots.append(new_root)
        except ValueError:
            pass
    if not roots:
        print("No Roots!")
        return roots
    else:
        return roots


def find_peaks(interdata, interderiv2, roots):
    peaks = []
    for item in roots:
        if interderiv2(item) < 0:
            peaks.append(item)
    return peaks


def data_report(file_list):
    answers = pd.DataFrame(columns=["Name", "Offset", "Radius", "Diameter"])

    # Numerical information about the spectra
    for filename in file_list:
        data = get_data(filename)
        offset = find_offset_wavelength(data, filename)
        energy = light_energy(offset)
        radius = find_radius(energy, 0.13, 0.45)

        data_dict = {"Name": filename.split('/')[-1], "Offset": offset, "Radius": radius, "Diameter": radius * 2}
        answers = answers.append(data_dict, ignore_index=True)

    return answers


def gaussian_report(filename):
    # Interpreted data about the spectra.
    # Separated due to excesive calculation time.

    data = get_data(filename)
    xdata = data["Wavelength"]
    ydata = data["Absorbance"]

    interdata = si.interp1d(xdata, ydata, kind=2)

    inf_peak = inf_peak_pairs(xdata, ydata)
    opt_guess = gaussian_approximations(inf_peak, xdata, ydata, interdata, filename)

    show_gaussians(xdata, ydata, opt_guess)
    plt.savefig(f"Plots/{filename[5:-4]}_G.png", dpi=300)
    plt.close()


def save_plots(filename):
    data = get_data(filename)
    xdata = data["Wavelength"]
    ydata = data["Absorbance"]

    plt.plot(xdata, ydata)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.savefig(f"Plots/{filename[5:-4]}.png", dpi=300)
    plt.close()


def main():
    dir = "Data/"  # This is the working directory
    files = glob.glob(dir + "*.csv")  # Find all the csv files in the working directory

    print("Data Started")
    numbers = data_report(files)
    numbers.to_csv("Final-Calculations.csv")
    print("Data Calculated\n")

    print("Plots Started")
    for filename in files:
        save_plots(filename)
    print("Plots Saved\n")

    print("Gaussians Started")
    for filename in files:
        print(f"Starting {filename}")
        gaussian_report(filename)
    print("Gaussians Saved")

    for file in files:
        data = get_data(file)
        xdata = data["Wavelength"]
        ydata = data["Absorbance"]
        plt.plot(xdata, ydata, label=str(file[10]))
    plt.legend()
    plt.savefig("Plots/Run_2_all.png", dpi=300)



if __name__ == "__main__":
    main()
