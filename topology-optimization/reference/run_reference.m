% Run the published reference codes on the benchmark set used in this example.
% Usage: octave-cli --no-gui run_reference.m <case>
args = argv(); which = args{1};
mkdir('out');
switch which
  case 'mbb'      % Andreassen et al. (2011), Sec. 3.4
    for ft = [1 2]
      top88_ref(60, 20, 0.5, 3, 2.4, ft);
      top88_ref(150, 50, 0.5, 3, 6.0, ft);
      top88_ref(300, 100, 0.5, 3, 16.0, ft);
    end
  case 'cantilever'  % Sigmund (2001) cantilever, run with the 88-line code
    for ft = [1 2]
      top88_cant_ref(32, 20, 0.4, 3, 1.2, ft);
    end
  case 'top3d'    % Liu and Tovar (2014) default example
    top3d_ref(60, 20, 4, 0.3, 3, 1.5);
  case 'lk'
    KE = lk_H8(0.3); dlmwrite('out/ref_lk_H8.txt', KE, 'precision', '%.15g');
end
