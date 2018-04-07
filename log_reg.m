clear; close all ; clc ;

combined_data=load('minescombineddata_tab_delimited.txt');
x=combined_data( :, [10]);
y=combined_data( :, [17]);
z=combined_data( :, [18]);
plotmatrix(y,z);
 