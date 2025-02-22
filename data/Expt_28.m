% 
% Experiment 28: Fourier Transform Spectroscopy
% Authors: Brian James & Bill Tango
%
% Rev. 1.1
% Original
%
% Rev. 1.2
% WJT: Fixed a typo and added the Hanning window.
%
% Rev. 1.3
% PT (SB): Fixed the Hanning window.
%
% This script is designed for use with the data generated by the
% Expt 28 software.
% The input file is assumed to be a tab-separated ASCII .txt file.
% The first line in the file is a header which will be ignored.
% Subsequent lines contain the data: 
%    OPD in nm 
%    Intensity in arbitrary units.
% If the number of data points is not a power of 2 padding 
% will be added to make it so and a Hanning window will be applied.
%

% Get the file name; quit if no file selected:
filename = uigetfile('*.txt', 'Choose file...');
if filename == 0 
    return
end
    
% Import the data into the structure 'input' and
% copy to the arrays x and y.
delimiterIn = '\t';
headerLinesIn = 1;
input = importdata(filename, delimiterIn, headerLinesIn);
x = input.data(:,1);
y = input.data(:,2);

% Plot the input data (x needs to be transposed):
figure;
plot(x', y, 'b');
xlabel('Optical Path Length (nm)', 'FontSize', 14);
ylabel('Intensity (a.u.)', 'FontSize', 14);

% Get the number of data points and the step size in nm
n_in = length(x);
opd_step = x(2) - x(1);

% Make the intensity data zero mean:
y_mean = mean(y);
yz = y - y_mean;

% Use a Hanning window function:
hwindow = hann(n_in, 'periodic');
yw = hwindow.*yz;

% Find the next larger power of 2:
n2 = 2;
while n_in > n2
    n2 = 2*n2;
end

% If necessary, pad the x and y vectors 
% (if n2 equals n_in no padding is needed).
if (n2 > n_in)
    for m = n_in+1:1:n2
        x(m) = x(m-1) + opd_step;
        yw(m) = 0;
    end
end

% Take the Fourier transform; take the positive range, normalise
% and take the modulus of the complex FFT:
transform = fft(yw);
transform_pos = transform(1:n2/2+1)*opd_step;
spectrum = abs(transform_pos);

% Calculate the abscissas (positive only) in the correct physical units
% Note the difference between 1.0/x and 1./x!!!
wnum_step = 1.0/opd_step;
wnum = wnum_step/n2*(0:n2/2);   % wavenumber vector
wlength = 1./wnum;              % wavelength vector      

% Select a sensible range of wavenumbers for plotting
wnum_plot = wnum(wnum>0.000833 & wnum<0.00333);
wlength_plot = 1./wnum_plot;    % the corresponding wavelength range
spectrum_plot = spectrum(wnum>0.000833 & wnum<0.00333);

% Plot the spectrum
figure;
plot(wlength_plot, spectrum_plot, 'b');
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Intensity (a.u.)', 'FontSize', 14);

% Save the FFT data
[path, name, ext] = fileparts(filename);
name = strcat(name, '_fft', ext);

% Format output array: 
%    row 1 is wavelength;
%    row2 is the spectrum (which needs to be transposed)
output = [wlength_plot; spectrum_plot'];

% Print output to file
fid = fopen(name, 'w');
fprintf(fid, '%6.2f\t%6.2f\n', output);
fclose(fid);

% All done
return
