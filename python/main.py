from __init__ import *
from utils import *
import plots
import icp
import pose_graph as pg
from dataset import Dataset

if __name__ == '__main__':
    # create dataset
    dataset = Dataset()

    # create folders
    for sequence in dataset.sequences:
        T_gt = dataset.get_data(sequence)
        for n in range(T_gt.shape[0]-1):
            path = os.path.join('results', sequence, str(n))
            if not os.path.exists(path):
                os.makedirs(path)

    # first loop over sequence
    for sequence in dataset.sequences:

        T_gt = dataset.get_data(sequence)

        # compute ICP covariance and results
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            # Censi covariance
            icp.censi(dataset, sequence, scan_ref, scan_in)
            # Monte-Carlo covariance
            icp.mc(dataset, sequence, scan_ref, scan_in)
            # Unscented transform covariance
            icp.ut(dataset, sequence, scan_ref, scan_in, Param.cov_ut)

        # get ICP results
        for scan_ref in range(T_gt.shape[0]-1):
            scan_in = scan_ref + 1
            # compute results
            icp.results(dataset, sequence, scan_ref, scan_in, Param.cov_ut)
            if Param.b_cov_plot:
                # plot results
                plots.cov_plot(dataset, sequence, scan_ref, scan_in)

    # second loop over sequences
    for sequence in dataset.sequences:
        # compute ICP for pose-graph
        pg.pg_icp(dataset, sequence)
        # compute pose graph results
        pg.compute(dataset, sequence)

        # get pose graph results
        plots.pg_plot(dataset, sequence)

    # display results
    icp.aggregate_results(dataset)
    pg.aggregate_results(dataset)



