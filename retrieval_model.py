import json
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def correlation_coefficient(x, y):
    # Calculate mean
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate different value
    diff_x = x - mean_x
    diff_y = y - mean_y

    # Calculate squart different value
    diff_x_sq = diff_x ** 2
    diff_y_sq = diff_y ** 2

    # Calculate the sum of products
    sum_xy = np.sum(diff_x * diff_y)

    # Calculate correlation coefficient
    correlation = sum_xy / (np.sqrt(np.sum(diff_x_sq)) * np.sqrt(np.sum(diff_y_sq)))

    return correlation


def data_qc_filter(dir_s, dir_q, rrs_qc_flag_dir, water_parameter_qc_flag_dir):
    """
        This function will remove data that does not meet quality control requirements.

        Args:
            input file:
            dir_s: remote sensing reflectance dataset
            dir_q: water parameter dataset
            rrs_qc_flag_dir: quality control of remote sensing reflectance file
            water_parameter_qc_flag_dir: quality control of water parameter dataset file
        Returns:
            rrs_qc(DataFrame): Dataset after removing remote sensing reflectance data
                               that does not meet quality control requirements
            wp_qc(DataFrame): Dataset after removing water parameter data
                              that does not meet quality control requirements
            No return file and variables
    """

    csv_s = pd.read_csv(dir_s)
    csv_q = pd.read_csv(dir_q)
    qc_s = pd.read_csv(rrs_qc_flag_dir)
    qc_q = pd.read_csv(water_parameter_qc_flag_dir)
    number = len(csv_s)
    normal_data = []
    unusual_data = []
    for i in range(number):
        flag = qc_s.iloc[i, 1:].sum()
        if flag == 0:
            normal_data.append(i)
            for j in range(1, 5):
                if qc_q.iloc[i, j] > 0:
                    csv_q.iloc[i, j + 4] = np.nan
        else:
            unusual_data.append(i)

    rrs_qc = csv_s.iloc[normal_data]
    wq_qc = csv_q.iloc[normal_data]
    print('Normal rrs data number:', len(normal_data))
    print('Unusual rrs data number:', len(unusual_data))
    return rrs_qc, wq_qc


def srf_function(wave, band, srf):
    """
        The remote sensing reflectance data is converted into water equivalent spectrum data from the HJ2A satellite.
        The formula for this function is in formula (3) of this manuscript.
        Args:
            input file:
            wave: wavelength
            band: Spectral data corresponding to wavelength
            srf:  the spectral response function
        Returns:
            wes(list): water equivalent spectrum data
    """
    min_x = np.min(wave)
    ind1 = 400 - min_x
    wes = []
    for SRF_ind in srf.keys():
        t1 = 0
        t2 = 0
        for i in range(ind1, len(band)):
            t2 = t2 + srf[SRF_ind][i]
            t1 = t1 + band[i] * 3.1415926 * srf[SRF_ind][i]
        wes.append(round(t1 / t2, 5))
    return wes


def wes_transfrom(rrs_qc):
    """
        The remote sensing reflectance data is converted into water equivalent spectrum data from the HJ2A satellite.
        Args:
            rrs_qc: Remote sensing reflectance data after quality control
        Returns:
            rrs_wes(DataFrame): all water equivalent spectrum data in dataset
    """
    satellite = 'HJ2A'
    sensor = 'CCD1'
    srf_path = './SRF/' + satellite + '.json'
    with open(srf_path, "r") as json_file:
        srf_data = json.load(json_file)
        srf = srf_data[sensor]['SRF']

    column_labels = rrs_qc.columns.tolist()
    wavelength = column_labels[5:]
    wave = [int(i) for i in wavelength]
    spectrum = []
    for i in range(len(rrs_qc)):
        sp = rrs_qc.iloc[i, 5:]
        sp = sp.fillna(0)
        temp = srf_function(wave, sp, srf)
        spectrum.append(temp)
    col = [satellite + '_' + str(j + 1) for j in range(5)]
    rrs_wes = pd.DataFrame(spectrum, columns=col, index=rrs_qc.index)
    rrs_wes.insert(loc=rrs_wes.columns.get_loc(col[0]), column='OID', value=rrs_qc['OID'])
    return rrs_wes


def retrieval_model(X, Y, parameter):
    """
        This function build chla retrieval model.
        Args:
            X (DataFrame): all water equivalent spectrum data in dataset
            Y (DataFrame): water parameter data after quality control
        Returns:
            rrs_wes(DataFrame): all water equivalent spectrum data in dataset
    """

    # divide train dataset and test dataset
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(X, Y, test_size=0.3, random_state=15)

    # delete nan value
    x_train = x_train_o[~np.isnan(y_train_o)]
    y_train = y_train_o[~np.isnan(y_train_o)]
    x_test = x_test_o[~np.isnan(y_test_o)]
    y_test = y_test_o[~np.isnan(y_test_o)]

    # Random_Forest_Regressor
    rf_regressor = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=3)
    rf_regressor.fit(x_train, y_train)
    with open('./Data/' + parameter[0:3] + '_random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_regressor, f)

    y_train_fit = rf_regressor.predict(x_train)
    y_test_fit = rf_regressor.predict(x_test)

    return y_train.values, y_test.values, y_train_fit, y_test_fit


def validation_plot(y, y_fit, label, parameter):
    """
        This function build chla retrieval model.
        Args:
            y (Numpy): Measured data
            y_fit (Numpy): Derived data
            label (str): a label 'test dataset' or 'Train dataset'
            parameter (str): water parameter name and unit
        Returns:
            No return
    """

    plt.figure(figsize=(4.5, 4.5))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    r = np.around(correlation_coefficient(y, y_fit), 3)
    mape = np.around(mean_absolute_percentage_error(y, y_fit), 3)
    rmse = np.around(np.sqrt(mean_squared_error(y, y_fit)), 3)
    print('R2 = ', r)
    print('MAPE = ', mape)
    print('RMSE = ', rmse)

    # Fitting straight line
    max_value = np.max([np.max(y), np.max(y_fit)])
    min_value = np.min([np.min(y), np.min(y_fit)])

    coefficients = np.polyfit(y, y_fit, 1)
    polynomial = np.poly1d(coefficients)
    x_ = np.linspace(min_value, max_value, 100)
    y_ = polynomial(x_)
    a = float(coefficients[0])
    b = float(coefficients[1])

    text2 = 'y=' + str(np.round(a, 2)) + 'x + ' + str(np.round(b, 2))
    text3 = text2 + '\n' + 'r  = ' + str(r) + '\n' + 'MAPE = ' + str(mape) + '\n' \
            + 'RMSE = ' + str(rmse)

    plt.plot([0, max_value * 1.05], [0, max_value * 1.05], 'k', linewidth=1, label='1:1')
    plt.scatter(y, y_fit, s=10, c='k', label=label)
    plt.plot(x_, y_, 'k', linewidth=1.5)

    # 设置图例
    fontdict = {'family': 'Times New Roman', 'size': 12, 'style': 'normal'}
    plt.text(max_value * 0.7, max_value * 0.04, text3, fontdict=fontdict)
    plt.xlabel('Measured ' + parameter, fontdict=fontdict)
    plt.ylabel('Derived ' + parameter, fontdict=fontdict)
    plt.xticks(fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.legend(loc='best', prop=fontdict, frameon=False)
    plt.axis([0, max_value * 1.05, 0, max_value * 1.05])
    plt.savefig('./picture/' + parameter[0:3] + '_' + label + '_data.png', dpi=300)
    plt.show()


def data_statistics(wq_qc):
    name = ['Chl-a', 'TSM', 'SDD', 'LWST']
    for i in range(4):
        print(name[i])
        q_qc = wq_qc.iloc[:, i + 5]
        data = q_qc[~np.isnan(q_qc)]
        max_value = data.max()
        min_value = data.min()
        mean_value = data.mean()
        std_value = data.std()
        print('max_value=', max_value)
        print('min_value=', min_value)
        print('mean_value=', mean_value)
        print('std_value=', std_value)


if __name__ == '__main__':
    # data loading
    # remote sensing reflectance data loading
    dir_s = r'./Data/Dataset/Remote sensing reflectance.csv'  # data dir
    # water parameter data loading
    dir_q = r'./Data/Dataset/Water quality dataset.csv'  # data dir
    # quality data loading
    rrs_qc_flag_dir = r'./Data/Dataset/Rrs_qc_flag.csv'
    water_parameter_qc_flag_dir = r'./Data/Dataset/Water_parameter_qc_flag.csv'

    # data filter
    # This function will remove data that does not meet quality control requirements
    rrs_qc, wq_qc = data_qc_filter(dir_s, dir_q, rrs_qc_flag_dir, water_parameter_qc_flag_dir)

    # The remote sensing reflectance data is converted into water equivalent spectrum data from the HJ2A satellite.
    rrs_wes = wes_transfrom(rrs_qc)

    # save water equivalent spectrum data
    rrs_wes_dir = r'./Data/water equivalent spectrum.csv'
    rrs_wes.to_csv(rrs_wes_dir, index=False)

    qc_dir = r'./Data/water parameter quality control data.csv'
    wq_qc.to_csv(qc_dir, index=False)

    data_statistics(wq_qc)

    # build Chl-a Random Forest model
    X = rrs_wes.iloc[:, 1:6]
    Y = wq_qc.iloc[:, 5]  # Chl-a data
    y_train, y_test, y_train_fit, y_test_fit = retrieval_model(X, Y, 'Chl-a(μg/L)')
    # plot function
    validation_plot(y_train, y_train_fit, 'Train dataset', 'Chl-a(μg/L)')
    validation_plot(y_test, y_test_fit, 'Test dataset', 'Chl-a(μg/L)')

    # build TSM Random Forest model
    X = rrs_wes.iloc[:, 1:6]
    Y = wq_qc.iloc[:, 6]  # TSM data
    y_train, y_test, y_train_fit, y_test_fit = retrieval_model(X, Y, 'TSM(mg/L)')
    # plot function
    validation_plot(y_train, y_train_fit, 'Train dataset', 'TSM(mg/L)')
    validation_plot(y_test, y_test_fit, 'Test dataset', 'TSM(mg/L)')

    # build SDD Random Forest model
    X = rrs_wes.iloc[:, 1:6]
    Y = wq_qc.iloc[:, 7]  # SDD data
    y_train, y_test, y_train_fit, y_test_fit = retrieval_model(X, Y, 'SDD(cm)')
    # plot function
    validation_plot(y_train, y_train_fit, 'Train dataset', 'SDD(cm)')
    validation_plot(y_test, y_test_fit, 'Test dataset', 'SDD(cm)')



