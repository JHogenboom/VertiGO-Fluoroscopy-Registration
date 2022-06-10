Will contain the generated comma delimited files (.csv) and scaleable vector graphics (.svg).

-----------------

The comma delimited files contain information on the digitally reconstructed radiograph and the image similarity score.

The filename will consist of '**PatientID**_evaluated_parameters_**3DModel**_filter_**ImageFilter**.csv'.

The contents will consist of Index,filename,Rotation x,Rotation y,Rotation z,Translation x,Translation y,Translation z,content loss

-----------------

The scaleable vector graphics files contain a plot of the image similarity produced by MatplotLib.

The filename will consist of '**PatientID 3DModel** - Preoperative - **ImageFilter** DRR n-D landscape - Intraoperative - **ImageFilter** of **FluoroscopyImage**.svg'.

