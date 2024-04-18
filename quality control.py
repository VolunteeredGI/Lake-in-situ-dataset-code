import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rrs_qc(dir_s):
    """
        Add remote sensing reflectance data address(.csv file) and
        This function will output the quality control results(.csv file).  .

        Args:
            dir_q: input file
                    example: dir_s = r'./Data/Dataset/Remote sensing reflectance.csv'
        Returns:
            No return file and variables
    """
    csv_s = pd.read_csv(dir_s)
    index = csv_s['OID']

    rrs_flag = pd.DataFrame(index=index, columns=['Max_value', 'Min_value', 'Noisy_red',
                                                  'Noisy_blue', 'Baseline_shift', 'Negative_uv_slope',
                                                  'QWIP_fail', ])
    ancillary = pd.DataFrame(index=index, columns=['Apparent_visible_wavelength', 'QWIP_score'])
    rrs_flag.fillna(0, inplace=True)
    ancillary.fillna(0, inplace=True)

    for i in range(len(csv_s)):
        # read spectrum data
        spectrum = csv_s.iloc[i, 5:]
        spectrum = spectrum.fillna(0)
        wave = [int(k) for k in spectrum.index.values]
        # max value
        spectrum_max_value = spectrum.max()
        # min value
        spectrum_min_value = spectrum.min()
        # mean value
        spectrum_median_value = spectrum.median()

        if spectrum_max_value > 0.2:
            rrs_flag.iloc[i, 0] = 1
        elif spectrum_min_value < 0:
            rrs_flag.iloc[i, 1] = 1

        # spectra were standardized to zero mean and unit standard deviation
        X = spectrum.values.reshape(-1, 1)
        scaler = StandardScaler()
        spectrum_scaled = scaler.fit_transform(X)

        # Noisy_red calculation
        spectrum_scaled_750_940 = spectrum_scaled[400:]

        x = np.array(wave[400:]).reshape(-1)
        y = spectrum_scaled_750_940.reshape(-1)

        coefficients = np.polyfit(x, y, 4)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(x[0], x[-1], x.shape[0])
        y_fit = polynomial(x_fit)

        rmse = np.sqrt(np.mean((y_fit - y) ** 2))
        Noisy_red = rmse

        if Noisy_red > 0.2:
            rrs_flag.iloc[i, 2] = 1

        # Noisy_blue calculation
        spectrum_scaled_350_400 = spectrum_scaled[0:51]

        x = np.array(wave[0:51]).reshape(-1)
        y = np.array(spectrum_scaled_350_400).reshape(-1)

        coefficients = np.polyfit(x, y, 4)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(x[0], x[-1], x.shape[0])
        y_fit = polynomial(x_fit)

        rmse = np.sqrt(np.mean((y_fit - y) ** 2))
        Noisy_blue = rmse

        if Noisy_blue > 0.15:
            rrs_flag.iloc[i, 3] = 1

        # Baseline_shift calculation
        threshold = spectrum_median_value * 0.6
        if spectrum_min_value >= threshold:
            rrs_flag.iloc[i, 4] = 1

        negative_count = spectrum[spectrum < 0].size
        if negative_count > 20:
            rrs_flag.iloc[i, 4] = 2

        spectrum_765_940 = spectrum[415:]
        spectrum_350_450 = spectrum[0:101]
        q1 = np.percentile(spectrum, 25)  # First quartile (lower quartile)

        x = np.array(wave[415:]).reshape(-1)
        y = spectrum_765_940.values.reshape(-1).astype(float)
        coefficients = np.polyfit(x, y, 1)
        slope = float(coefficients[0])
        negative_count_765_940 = spectrum_765_940[spectrum_765_940 < 0].size
        negative_count_350_450 = spectrum_350_450[spectrum_350_450 < 0].size
        if slope < q1 and negative_count_765_940 > 0.5 * (940 - 750):
            rrs_flag.iloc[i, 4] = 3
        if negative_count_765_940 > 0.7 * (940 - 750):
            rrs_flag.iloc[i, 4] = 4
        if negative_count_350_450 > 20:
            rrs_flag.iloc[i, 4] = 5

        # Negative_uv_slope calculation
        spectrum_350_420 = spectrum[0:71]
        x = np.array(wave[0:71]).reshape(-1)
        y = spectrum_350_420.values.reshape(-1).astype(float)
        coefficients = np.polyfit(x, y, 1)
        slope = float(coefficients[0])
        if slope < -0.005:
            rrs_flag.iloc[i, 5] = 1

        # QWIP_fail calculation
        spectrum_sum_value = spectrum.sum()
        temp = 0
        for k in range(len(spectrum)):
            temp = temp + spectrum[k] / wave[k]

        # Apparent_visible_wavelength
        avw = spectrum_sum_value / temp
        ancillary.iloc[i, 0] = avw
        # Normalized Difference Index
        ndi = (spectrum[314] - spectrum[141]) / (spectrum[314] + spectrum[141])
        QWIP = ((3.54951959 * 10 ** (-9)) * avw ** 4 - (8.26904176 * 10 ** (-6)) * avw ** 3 +
                (7.13990716 * 10 ** (-3)) * avw ** 2 - 2.70244824 * avw + 3.77349867 * 10 ** 2)
        QWIP_score = ndi - QWIP
        ancillary.iloc[i, 1] = QWIP_score
        if np.abs(QWIP_score) > 0.2:
            rrs_flag.iloc[i, 6] = 1

    # write rrs_flag and ancillary data
    rrs_flag_dir = r'./Data/Dataset/rrs_qc_flag.csv'
    rrs_flag.to_csv(rrs_flag_dir)
    ancillary_dir = r'./Data/Dataset/rrs_qc_ancillary.csv'
    ancillary.to_csv(ancillary_dir)


def water_parameter_qc(dir_q):
    """
        Add water parameter data address(.csv file) and
        This function will output the quality control results(.csv file).  .

        Args:
            dir_q: input file
            example: dir_s = r'./Data/Dataset/Water quality dataset.csv'
        Returns:
            No return file and variables
    """
    csv_q = pd.read_csv(dir_q)
    # new Create a new quality control variable
    index = csv_q['OID']
    flag = pd.DataFrame(index=index, columns=['Chl-a_flag', 'TSM_flag', 'SDD_flag', 'LWST_flag'])
    flag.fillna(0, inplace=True)

    # Chl-a quality control
    for i in range(len(csv_q)):
        chl_a = csv_q.iloc[i, 5]
        if chl_a >= 200:
            flag.iloc[i, 0] = 1
        elif chl_a <= 0:
            flag.iloc[i, 0] = 2
        elif np.isnan(chl_a):
            flag.iloc[i, 2] = 3

    # TSM quality control
    for i in range(len(csv_q)):
        tsm = csv_q.iloc[i, 6]
        if tsm >= 500:
            flag.iloc[i, 1] = 1
        elif tsm <= 0:
            flag.iloc[i, 1] = 2
        elif np.isnan(tsm):
            flag.iloc[i, 2] = 3

    # SDD quality control
    for i in range(len(csv_q)):
        sdd = csv_q.iloc[i, 7]
        if sdd >= 500:
            flag.iloc[i, 2] = 1
        elif sdd <= 0:
            flag.iloc[i, 2] = 2
        elif np.isnan(sdd):
            flag.iloc[i, 2] = 3

    # LWST quality control
    for i in range(len(csv_q)):
        lwst = csv_q.iloc[i, 8]
        if lwst >= 40:
            flag.iloc[i, 3] = 1
        elif lwst <= 0:
            flag.iloc[i, 3] = 2
        elif np.isnan(lwst):
            flag.iloc[i, 2] = 3

    # write flag data
    dir_flag = r'./Data/Dataset/Water_parameter_qc_flag.csv'
    flag.to_csv(dir_flag)


if __name__ == '__main__':
    # rrs quality control funtion
    dir_s = r'./Data/Dataset/Remote sensing reflectance.csv'  # data dir
    rrs_qc(dir_s)

    # water parameter quality control function
    dir_q = r'./Data/Dataset/Water quality dataset.csv'  # data dir
    water_parameter_qc(dir_q)