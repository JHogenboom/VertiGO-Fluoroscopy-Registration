#  PYTHON MODULES
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
    Module with a plethora of small functions that built plots with a consistent style
    
    Written by Joshi Hogenboom in Januari 2022.
"""


def insert_three_dimensional_aesthetics(ax):

    # force z-axis to the left for consistency and colourbar
    # credit to crayzeewulf
    # https://stackoverflow.com/questions/15042129/changing-position-of-vertical-z-axis-of-3d-plot-matplotlib
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])

    # plot view
    ax.elev = 19.53

    # remove grid and backdrop
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def save_plot(ax, filename, dimensions):
    """
        Applies overall aesthetics, adapts filename and saves the plot

        :param ax: pyplot figure subplot
        :param str filename: filename
        :param int dimensions: number of orientation parameters with variation
    """
    # save plot as scalable vector graphic file using adapted title
    ax.tick_params(axis='both', colors='grey')
    filename = filename.replace('n-D', f'{dimensions}D')
    plt.savefig(f'{filename}.svg', aspect='auto')


def insert_title(ax, plot_title, dimensions):
    """
        Inserts string with orientation parameters that has the highest similarity i.e., lowest loss
        The string is added similarly to a subtitle

        :param ax: pyplot figure subplot
        :param str plot_title: plot title
        :param int dimensions: number of orientation parameters with variation
    """
    # aesthetics and produce correct title
    plot_title = plot_title.replace('n-D', f'{dimensions}D')
    ax.set_title(plot_title, pad=20)


def insert_axis(plot_axes, axis_number):
    """
        Retrieve and format axis label and ticks for an axis containing orientation parameters

        :param dict plot_axes: dictionary containing axis points and labels
        :param int axis_number: index of axis to be returned
        :return numpy.ndarray axis_points: ticks for axis
        :return str label: label of axis
    """
    # extract axis points for axis
    axis_points = plot_axes['Points'][axis_number]
    axis_points = axis_points.astype(np.float)

    # determine name for axis using the name associated with the axis points
    label = plot_axes['Axis'][axis_number]
    label = label[2: label.index("']")]

    # add appropriate units
    if 'Rotation' in label:
        label = f'{label}\nin degrees'
    elif 'Translation' in label:
        label = f'{label}\nin millimetres'

    return axis_points, label


def insert_match_string(ax, data_array, loss, dimensions):
    """
        Inserts string with orientation parameters that has the highest similarity i.e., lowest loss
        The string is added similarly to a subtitle

        :param ax: pyplot figure subplot
        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :param list loss: the loss determined for each sample
        :param int dimensions: number of orientation parameters with variation
    """
    # find minimum and associated parameters
    match_index = np.where(loss == np.amin(loss))[0] + 1
    match_parameters = str(
        f"Parameters associated with highest similarity: "
        f"rotation x: {data_array[match_index, 2]} y: {data_array[match_index, 3]} z: {data_array[match_index, 4]} "
        f"translation x: {data_array[match_index, 5]} y: {data_array[match_index, 6]} z: {data_array[match_index, 7]}")

    # remove brackets and apostrophes for aesthetic reasons
    match_parameters = match_parameters.replace("['", '')
    match_parameters = match_parameters.replace("']", '')

    # insert text underneath title
    if dimensions == 1:
        ax.text(0.5, 1.02, match_parameters, transform=ax.transAxes, fontsize=9, color='grey', ha='center')
    elif dimensions > 1:
        ax.text2D(0.5, 0.94, match_parameters, transform=ax.transAxes, fontsize=9, color='grey', ha='center')


def insert_colourbar(figure, ax, plot):
    """
        Inserts colourbar on the right side of the figure

        :param figure: pyplot figure
        :param ax: pyplot figure subplot
        :param plot: data in plot
    """
    # add colourbar
    clb = figure.colorbar(plot, ax=ax)

    # layout colourbar
    clb.ax.ticklabel_format(style='plain')
    clb.ax.set_title('Loss', rotation=0)


def format_axes(data_array):
    """
        Determines the unique values per axis and collects the corresponding axis label

        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :return dict axes: dictionary containing axis points and labels
    """
    # ensure that the correct number of axis is extracted for data analysis
    axes_for_plot = np.delete(data_array, 0, 1)

    # remove filenames from array
    axes_for_plot = np.delete(axes_for_plot, 0, 1)

    # extract the unique axis points per parameter and store in dictionary if larger than one
    axes = {'Axis': [], 'Points': []}
    for axis in range(6):
        axis_points = pd.unique(axes_for_plot[1:, axis])
        if len(axis_points) > 1:
            axes['Axis'].append(str(axes_for_plot[:1, axis]))
            axes['Points'].append(np.array(axis_points))
    return axes


def select_plot(plot_title, plot_axes, data_array, filename):
    """
        Selects the plot type suitable for the variation found in the axes

        :param str plot_title: plot title
        :param dict plot_axes: dictionary containing axis points and labels
        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :param str filename: path and filename
    """

    if len(plot_axes['Axis']) == 0:
        exit('No parameter variation was detected in evaluated parameters file.\nExiting program.')
    elif len(plot_axes['Axis']) == 1:
        one_dimensional_plot(plot_title, plot_axes, data_array, filename)
    elif len(plot_axes['Axis']) == 2:
        two_dimensional_plot(plot_title, plot_axes, data_array, filename)
    elif len(plot_axes['Axis']) == 3:
        three_dimensional_plot(plot_title, plot_axes, data_array, filename)
    elif len(plot_axes['Axis']) >= 4:
        exit('The number of dimensions required to visualise the variation in parameters is not supported.\n'
             'Exiting program.')


def one_dimensional_plot(plot_title, plot_axes, data_array, filename):
    """
        Generates a two-dimensional plot using Matplotlib and saves it under defined filename
        For more information on used functions:
        https://matplotlib.org/stable/api/pyplot_summary.html

        :param str plot_title: plot title
        :param dict plot_axes: dictionary containing axis points and labels
        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :param str filename: path and filename
    """

    # retrieve data points and label x-axis
    x_data, x_label = insert_axis(plot_axes, 0)

    # extract feature loss and store as float for y-axis
    y_loss = data_array[1:, 8]
    y_loss = y_loss.astype(np.float)

    # plot axis points on x, feature loss on y
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot()
    plot = ax.scatter(x_data, y_loss, c=y_loss, cmap='cividis')

    # layout and aesthetics x-axis
    ax.set_xlabel(x_label, labelpad=8, color='grey')
    ax.set_xticks(x_data)
    plt.locator_params(axis='x', nbins=12)

    # layout and aesthetics y-axis
    ax.set_ylabel('Relative\nsimilarity', rotation='horizontal', color='grey')
    ax.set_yticks([min(y_loss), max(y_loss)])
    ax.set_yticklabels(['High', 'Low'])

    insert_title(ax, plot_title, 1)
    insert_colourbar(fig, ax, plot)
    insert_match_string(ax, data_array, y_loss, 1)

    # 2D specific aesthetics
    sns.despine()

    # apply tick aesthetic and save
    save_plot(ax, filename, 1)


def two_dimensional_plot(plot_title, plot_axes, data_array, filename):
    """
        Generates a three-dimensional plot using Matplotlib and saves it under defined filename
        For more information on used functions:
        https://matplotlib.org/stable/api/pyplot_summary.html
        https://matplotlib.org/stable/api/axes_api.html

        :param str plot_title: plot title
        :param dict plot_axes: dictionary containing axis points and labels
        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :param str filename: path and filename
    """
    # retrieve data points and label x-axis
    x_axis_points, x_label = insert_axis(plot_axes, 0)

    # retrieve data for x-axis
    axes = list(data_array[0, :])
    x_data_position = axes.index(x_label[0: x_label.index('\n')])
    x_data = data_array[1:, x_data_position].astype(np.float)

    # retrieve data points and label y-axis
    y_axis_points, y_label = insert_axis(plot_axes, 1)

    # retrieve data for y-axis
    axes = list(data_array[0, :])
    y_data_position = axes.index(y_label[0: y_label.index('\n')])
    y_data = data_array[1:, y_data_position].astype(np.float)

    # extract feature loss and store as float for z-axis
    z_loss = data_array[1:, 8]
    z_loss = z_loss.astype(np.float)

    # plot axis points on x, axis points on y, feature loss on z
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection='3d')
    plot = ax.scatter3D(x_data, y_data, z_loss, c=z_loss, cmap='plasma')

    # layout and aesthetics x-axis
    ax.set_xlabel(x_label, labelpad=8, color='grey')
    ax.set_xticks(x_axis_points)
    plt.locator_params(axis='x', nbins=12)

    # layout and aesthetics y-axis
    ax.set_ylabel(y_label, labelpad=8, color='grey')
    ax.set_yticks(y_axis_points)
    plt.locator_params(axis='y', nbins=6)

    # layout and aesthetics z-axis
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Relative\nsimilarity', rotation='horizontal', color='grey')
    ax.set_zticks([min(z_loss), max(z_loss)])
    ax.set_zticklabels(['High', 'Low'])

    # global aesthetics
    insert_title(ax, plot_title, 2)
    insert_colourbar(fig, ax, plot)
    insert_match_string(ax, data_array, z_loss, 2)

    # 3D specific aesthetics
    insert_three_dimensional_aesthetics(ax)

    # apply tick aesthetic and save
    save_plot(ax, filename, 2)


def three_dimensional_plot(plot_title, plot_axes, data_array, filename):
    """
        Generates a three-dimensional plot using Matplotlib and saves it under defined filename
        For more information on used functions:
        https://matplotlib.org/stable/api/pyplot_summary.html
        https://matplotlib.org/stable/api/axes_api.html

        :param str plot_title: plot title
        :param dict plot_axes: dictionary containing axis points and labels
        :param numpy.ndarray data_array: landscape output array; n x 8 array as str
        :param str filename: path and filename
    """
    # remove data TODO
    # max_loss = np.max(data_array[1:, 8])
    # print(max_loss)
    # data_array[1:, 8]

    # extract axis points for x-axis
    x_axis_points = plot_axes['Points'][0]
    x_axis_points = x_axis_points.astype(np.float)

    # determine name for x-axis using the name associated with the axis points
    x_label = plot_axes['Axis'][0]
    x_label = x_label[2: x_label.index("']")]

    # retrieve data for x-axis
    axes = list(data_array[0, :])
    x_data_position = axes.index(x_label)
    x_data = data_array[1:, x_data_position].astype(np.float)

    # extract axis points for y-axis
    y_axis_points = plot_axes['Points'][1]
    y_axis_points = y_axis_points.astype(np.float)

    # determine name for y-axis using the name associated with the axis points
    y_label = plot_axes['Axis'][1]
    y_label = y_label[2: y_label.index("']")]

    # retrieve data for y-axis
    axes = list(data_array[0, :])
    y_data_position = axes.index(y_label)
    y_data = data_array[1:, y_data_position].astype(np.float)

    # extract axis points for z-axis
    z_axis_points = plot_axes['Points'][1]
    z_axis_points = z_axis_points.astype(np.float)

    # determine name for z-axis using the name associated with the axis points
    z_label = plot_axes['Axis'][1]
    z_label = z_label[2: z_label.index("']")]

    # retrieve data for z-axis
    axes = list(data_array[0, :])
    z_data_position = axes.index(z_label)
    z_data = data_array[1:, z_data_position].astype(np.float)

    # extract feature loss and store as float for z-axis
    loss = data_array[1:, 8]
    loss = loss.astype(np.float)

    # plot axis points on x, axis points on y, axis points on z, feature loss in colourbar
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection='3d')
    plot = ax.scatter3D(x_data, y_data, z_data, c=loss, cmap='cividis')

    # layout and aesthetics overall
    plot_title = plot_title.replace('n-D', '3D')
    ax.set_title(plot_title, pad=20)

    # layout and aesthetics x-axis
    ax.set_xlabel(x_label, labelpad=8, color='grey')
    ax.set_xticks(x_axis_points)
    plt.locator_params(axis='x', nbins=12)

    # layout and aesthetics y-axis
    ax.set_ylabel(y_label, labelpad=8, color='grey')
    ax.set_yticks(y_axis_points)
    plt.locator_params(axis='y', nbins=6)

    # force z-axis to the left for consistency and colourbar
    # credit to crayzeewulf
    # https://stackoverflow.com/questions/15042129/changing-position-of-vertical-z-axis-of-3d-plot-matplotlib
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])

    # layout and aesthetics z-axis
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(z_label, rotation='horizontal', color='grey')
    ax.set_zticks(z_axis_points)
    plt.locator_params(axis='z', nbins=6)

    # layout colourbar
    clb = fig.colorbar(plot, ax=ax)
    clb.ax.ticklabel_format(style='plain')
    clb.ax.set_title('Loss', rotation=0, color='grey')

    # plot view and aesthetics
    ax.elev = 19.53
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # save plot as scalable vector graphic file using plot title
    ax.tick_params(axis='both', colors='grey', length=0)
    filename = filename.replace('n-D', '3D')
    plt.savefig(f'{filename}.svg', aspect='auto')
