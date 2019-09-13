from __init__ import *
from utils import *
import plots


def censi(dataset, sequence, scan_ref, scan_in):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    pose_path = os.path.join(base_path, "T_censi.txt")
    cov_path = os.path.join(base_path, "cov_censi.txt")
    if not Param.b_cov_icp and os.path.exists(pose_path):
        print(pose_path + " already exist")
        return
    T_gt = dataset.get_data(sequence)
    pc_ref = dataset.get_pc(sequence, scan_ref)
    pc_in = dataset.get_pc(sequence, scan_in)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    icp_with_cov(pc_ref, pc_in, T_init, Param.config_yaml, pose_path, cov_path)


def mc(dataset, sequence, scan_ref, scan_in):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    path = os.path.join(base_path, "mc_" + str(Param.n_mc-1) + ".txt")
    if not Param.b_cov_icp and os.path.exists(path):
        print(path + " already exist")
        return
    T_gt = dataset.get_data(sequence)
    pc_ref = dataset.get_pc(sequence, scan_ref)
    pc_in = dataset.get_pc(sequence, scan_in)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])

    # Monte-Carlo
    for n in range(Param.n_mc):
        path = os.path.join(base_path, "mc_" + str(n) + ".txt")
        if not Param.b_cov_icp and os.path.exists(path):
            print(path + " already exist")
            continue

        # sample initial transformation
        xi = np.hstack((np.random.normal(0, Param.cov_std_rot, 3),
                        np.random.normal(0, Param.cov_std_pos, 3)))
        T_init_n = SE3.normalize(SE3.mul(SE3.exp(-xi), T_init))  # T = exp(xi) T_hat
        icp_without_cov(pc_ref, pc_in, T_init_n, path)


def ut(dataset, sequence, scan_ref, scan_in, ut):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    path = os.path.join(base_path, "ut_12.txt")
    if not Param.b_cov_icp and os.path.exists(path):
        print(path + " already exist")
        return
    T_gt = dataset.get_data(sequence)
    pc_ref = dataset.get_pc(sequence, scan_ref)
    pc_in = dataset.get_pc(sequence, scan_in)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])

    # sigma-points
    sps = ut.sp.sigma_points(ut.Q_prior)
    f_sp = os.path.join(base_path, "sp_sigma_points.txt")
    np.savetxt(f_sp, sps)
    # unscented transform
    for n in range(13):
        path = os.path.join(base_path, "ut_" + str(n) + ".txt")
        if not Param.b_cov_icp and os.path.exists(path):
            print(path + " already exist")
            continue
        T_init_n = SE3.normalize(SE3.mul(SE3.exp(sps[n]), T_init))  #  T_sp = exp(xi) T_hat
        icp_without_cov(pc_ref, pc_in, T_init_n, path)


def results(dataset, sequence, scan_ref, scan_in, ut_class):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    f_metrics = os.path.join(base_path, 'metrics.p')
    if not Param.b_cov_results and os.path.exists(f_metrics):
        print(f_metrics + " already exists")
        return
    cov_path = os.path.join(base_path, "cov_censi.txt")
    T_gt = dataset.get_data(sequence)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    T_mc, T_init_mc = dataset.get_mc_results(sequence, scan_ref)
    T_ut, T_init_ut = dataset.get_ut_results(sequence, scan_ref)

    _, _, cov_ut, cov_cross = ut_class.unscented_transform_se3(T_ut)

    cov_base = np.zeros((6, 6))
    for seq in dataset.sequences:
        b_path = os.path.join(Param.results_path, seq)
        f_base = os.path.join(b_path, 'base.p')
        cov_base += 1/8*dataset.load(f_base)['cov_base']

    # Monte-Carlo errors
    mc_new = np.zeros((Param.n_mc, 6))
    T_init_inv = SE3.inv(T_init)
    for n in range(Param.n_mc):
        mc_new[n] = SE3.log(SE3.mul(T_mc[n], T_init_inv))  # xi = log( T * T_hat^{-1} )
    cov_mc = np.cov(mc_new.T)

    mc_65_new_idxs = np.random.choice(np.arange(Param.n_mc), 65, replace=False)
    mc_65_new = mc_new[mc_65_new_idxs]
    cov_mc_65 = np.cov(mc_65_new.T)

    data = np.genfromtxt(cov_path)
    cov_censi = Param.std_sensor**2 * data[:6]
    cov_prop = Param.std_sensor**2 * data[6:] + cov_ut

    kl_div_censi = np.zeros((Param.n_mc, 2))
    kl_div_mc_65 = np.zeros((Param.n_mc, 2))
    kl_div_prop = np.zeros((Param.n_mc, 2))
    kl_div_base = np.zeros((Param.n_mc, 2))
    nne_censi = np.zeros((Param.n_m, 2))
    nne_mc_65 = np.zeros((Param.n_m, 2))
    nne_prop = np.zeros((Param.n_mc, 2))
    nne_base = np.zeros((Param.n_mc, 2))

    for n in range(Param.n_mc):
        kl_div_censi[n] = rot_trans_kl_div(cov_mc, cov_censi)
        kl_div_mc_65[n] = rot_trans_kl_div(cov_mc, cov_mc_65)
        kl_div_prop[n] = rot_trans_kl_div(cov_mc, cov_prop)
        kl_div_base[n] = rot_trans_kl_div(cov_mc, cov_base)

        nne_censi[n] = nne_rot_trans(mc_new[n], cov_censi)
        nne_mc_65[n] = nne_rot_trans(mc_new[n], cov_mc_65)
        nne_prop[n] = nne_rot_trans(mc_new[n], cov_prop)
        nne_base[n] = nne_rot_trans(mc_new[n], cov_base)

    # get rid of worst and best quantiles
    seuil_up = int(0.9*Param.n_mc)
    seuil_low = Param.n_mc - seuil_up

    kl_div_censi = np.sort(kl_div_censi.T).T
    kl_div_mc_65 = np.sort(kl_div_mc_65.T).T
    kl_div_prop = np.sort(kl_div_prop.T).T
    kl_div_base = np.sort(kl_div_base.T).T

    kl_div_censi = np.sum(kl_div_censi[seuil_low:seuil_up], 0)
    kl_div_mc_65 = np.sum(kl_div_mc_65[seuil_low:seuil_up], 0)
    kl_div_prop = np.sum(kl_div_prop[seuil_low:seuil_up], 0)
    kl_div_base = np.sum(kl_div_base[seuil_low:seuil_up], 0)

    nne_censi = np.sort(nne_censi.T).T
    nne_mc_65 = np.sort(nne_mc_65.T).T
    nne_prop = np.sort(nne_prop.T).T
    nne_base = np.sort(nne_base.T).T

    nne_censi = np.sum(nne_censi[seuil_low:seuil_up], 0)
    nne_mc_65 = np.sum(nne_mc_65[seuil_low:seuil_up], 0)
    nne_prop = np.sum(nne_prop[seuil_low:seuil_up], 0)
    nne_base = np.sum(nne_base[seuil_low:seuil_up], 0)

    tmp = seuil_up - seuil_low
    kl_div_censi /= tmp
    kl_div_mc_65 /= tmp
    kl_div_prop /= tmp
    kl_div_base /= tmp
    nne_censi /= tmp
    nne_mc_65 /= tmp
    nne_prop /= tmp
    nne_base /= tmp

    metrics = {
        'kl_div_censi': kl_div_censi,
        'kl_div_mc_65': kl_div_mc_65,
        'kl_div_prop': kl_div_prop,
        'kl_div_base': kl_div_base,
        'nne_censi': nne_censi,
        'nne_mc_65': nne_mc_65,
        'nne_prop': nne_prop,
        'nne_base': nne_base,
        'cov_censi': cov_censi,
        'cov_prop': cov_prop,
        'cov_ut': cov_ut,
        'cov_cross': cov_cross,
        'cov_mc': cov_mc,
        'cov_mc_65': cov_mc_65,
        'T_true': T_init,
        'T_mc': T_mc,
        'T_init_mc': T_init_mc,
        'cov_base': cov_base,
        }
    dataset.dump(metrics, f_metrics)


def results_latex(dataset, sequence, scan_ref):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    f_metrics = os.path.join(base_path, 'metrics.p')
    metrics = dataset.load(f_metrics)

    T_mc = metrics['T_mc']
    T_init_mc = metrics['T_init_mc']
    T_true = metrics['T_true']

    cov_censi = metrics['cov_censi']
    cov_prop = metrics['cov_prop']
    cov_mc_65 = metrics['cov_mc_65']

    file_name = os.path.join(Param.latex_path, sequence + str(scan_ref) + 'mc.txt')
    header = "x_mc y_mc x_mc_init y_mc_init"
    data = np.zeros((T_mc.shape[0], 4))
    data[:, :2] = T_mc[:, :2, 3]
    data[:, 2:] = T_init_mc[:, :2, 3]
    np.savetxt(file_name, data, comments='', header=header)
    file_name = os.path.join(Param.latex_path, sequence + str(scan_ref) + 'T.txt')
    header = "x_true y_true"
    data = np.zeros((1, 2))
    data[0, 0] = T_true[0, 3]
    data[0, 1] = T_true[1, 3]
    np.savetxt(file_name, data, comments='', header=header)
    file_name = os.path.join(Param.latex_path, sequence + str(scan_ref) + 'Q.txt')
    header = "x_censi y_censi x_mc65 y_mc65 x_prop y_prop"

    xy_censi = plots.contour_ellipse(T_true, cov_censi, sigma=3, alpha=0.1)
    xy_prop = plots.contour_ellipse(T_true, cov_prop, sigma=3, alpha=0.1)
    xy_mc_65 = plots.contour_ellipse(T_true, cov_mc_65, sigma=3, alpha=0.1)

    n_min = np.min([150, xy_prop.shape[0], xy_censi.shape[0], xy_mc_65.shape[0]])
    xy_prop = xy_prop[np.linspace(0, xy_prop.shape[0]-1, n_min, dtype=int)]
    xy_censi = xy_censi[np.linspace(0, xy_censi.shape[0]-1, n_min, dtype=int)]
    xy_mc_65 = xy_mc_65[np.linspace(0, xy_mc_65.shape[0]-1, n_min, dtype=int)]

    data = np.zeros((xy_censi.shape[0], 6))
    data[:, :2] = xy_censi
    data[:, 2:4] = xy_mc_65
    data[:, 4:] = xy_prop
    np.savetxt(file_name, data, comments='', header=header)


def aggregate_results(dataset):
    kl_div_censi = np.zeros(2)
    kl_div_base = np.zeros(2)
    kl_div_mc_65 = np.zeros(2)
    kl_div_prop = np.zeros(2)
    nne_censi = np.zeros(2)
    nne_mc_65 = np.zeros(2)
    nne_prop = np.zeros(2)
    nne_base = np.zeros(2)
    n_tot = 0

    for sequence in dataset.sequences:
        T_gt = dataset.get_data(sequence)
        kl_div_censi_seq = np.zeros((T_gt.shape[0]-1, 2))
        kl_div_mc_65_seq = np.zeros((T_gt.shape[0]-1, 2))
        kl_div_prop_seq = np.zeros((T_gt.shape[0]-1, 2))
        kl_div_base_seq = np.zeros((T_gt.shape[0]-1, 2))
        nne_censi_seq = np.zeros((T_gt.shape[0]-1, 2))
        nne_mc_65_seq = np.zeros((T_gt.shape[0]-1, 2))
        nne_prop_seq = np.zeros((T_gt.shape[0]-1, 2))
        nne_base_seq = np.zeros((T_gt.shape[0]-1, 2))
        n_tot += T_gt.shape[0]-1

        cov_base = np.zeros((6, 6))
        for n in range(T_gt.shape[0]-1):
            base_path = os.path.join(Param.results_path, sequence, str(n))
            f_metrics = os.path.join(base_path, 'metrics.p')
            metrics = dataset.load(f_metrics)
            cov_base += metrics['cov_mc']
        cov_base /= T_gt.shape[0]-1
        metrics = {'cov_base': cov_base}
        base_path = os.path.join(Param.results_path, sequence)
        f_metrics = os.path.join(base_path, 'base.p')
        dataset.dump(metrics, f_metrics)

        for n in range(T_gt.shape[0]-1):
            base_path = os.path.join(Param.results_path, sequence, str(n))
            f_metrics = os.path.join(base_path, 'metrics.p')
            metrics = dataset.load(f_metrics)

            if sequence == dataset.sequences[0] and (n == 3 or n == 14):
                continue

            kl_div_censi_seq[n] = metrics['kl_div_censi']
            kl_div_mc_65_seq[n] = metrics['kl_div_mc_65']
            kl_div_prop_seq[n] = metrics['kl_div_prop']
            kl_div_base_seq[n] = metrics['kl_div_base']
            nne_censi_seq[n] = metrics['nne_censi']
            nne_mc_65_seq[n] = metrics['nne_mc_65']
            nne_prop_seq[n] = metrics['nne_prop']
            nne_base_seq[n] = metrics['nne_base']
            print(sequence, n, metrics['nne_prop'])

        seuil_up = int(1*(T_gt.shape[0]-1))
        seuil_low = T_gt.shape[0]-1 - seuil_up
        kl_div_censi_seq = np.sort(kl_div_censi_seq.T).T
        kl_div_mc_65_seq = np.sort(kl_div_mc_65_seq.T).T
        kl_div_prop_seq = np.sort(kl_div_prop_seq.T).T
        kl_div_base_seq = np.sort(kl_div_base_seq.T).T

        kl_div_censi_seq = np.sum(kl_div_censi_seq[seuil_low:seuil_up], 0)
        kl_div_mc_65_seq = np.sum(kl_div_mc_65_seq[seuil_low:seuil_up], 0)
        kl_div_prop_seq = np.sum(kl_div_prop_seq[seuil_low:seuil_up], 0)
        kl_div_base_seq = np.sum(kl_div_base_seq[seuil_low:seuil_up], 0)

        nne_censi_seq = np.sort(nne_censi_seq.T).T
        nne_mc_65_seq = np.sort(nne_mc_65_seq.T).T
        nne_prop_seq = np.sort(nne_prop_seq.T).T
        nne_base_seq = np.sort(nne_base_seq.T).T

        nne_censi_seq = np.sum(nne_censi_seq[seuil_low:seuil_up], 0)
        nne_mc_65_seq = np.sum(nne_mc_65_seq[seuil_low:seuil_up], 0)
        nne_prop_seq = np.sum(nne_prop_seq[seuil_low:seuil_up], 0)
        nne_base_seq = np.sum(nne_base_seq[seuil_low:seuil_up], 0)

        kl_div_censi += kl_div_censi_seq
        kl_div_mc_65 += kl_div_mc_65_seq
        kl_div_prop += kl_div_prop_seq
        kl_div_base += kl_div_base_seq
        nne_censi += nne_censi_seq
        nne_mc_65 += nne_mc_65_seq
        nne_prop += nne_prop_seq
        nne_base += nne_base_seq

        tmp = seuil_up - seuil_low
        kl_div_censi_seq /= tmp
        kl_div_mc_65_seq /= tmp
        kl_div_prop_seq /= tmp
        kl_div_base_seq /= tmp

        nne_censi_seq /= tmp
        nne_mc_65_seq /= tmp
        nne_prop_seq /= tmp
        nne_base_seq /= tmp

        nne_censi_seq = np.sqrt(nne_censi_seq)
        nne_mc_65_seq = np.sqrt(nne_mc_65_seq)
        nne_prop_seq = np.sqrt(nne_prop_seq)
        nne_base_seq = np.sqrt(nne_base_seq)

        # display results
        print('Covariance results for sequence ' + sequence)
        print('  Kullback-Leibler divergence')
        print('    -translation')
        print('      -Censi: {:.3f}'.format(kl_div_censi_seq[1]))
        print('      -Monte-Carlo 65: {:.3f}'.format(kl_div_mc_65_seq[1]))
        print('      -proposed: {:.3f}'.format(kl_div_prop_seq[1]))
        print('      -base: {:.3f}'.format(kl_div_base_seq[1]))
        print('    -rotation')
        print('      -Censi: {:.3f}'.format(kl_div_censi_seq[0]))
        print('      -Monte-Carlo 65: {:.3f}'.format(kl_div_mc_65_seq[0]))
        print('      -proposed: {:.3f}'.format(kl_div_prop_seq[0]))
        print('      -base: {:.3f}'.format(kl_div_base_seq[0]))
        print('  Normalized Norm Error')
        print('    -translation')
        print('      -Censi: {:.3f}'.format(nne_censi_seq[1]))
        print('      -Monte-Carlo 65: {:.3f}'.format(nne_mc_65_seq[1]))
        print('      -proposed: {:.3f}'.format(nne_prop_seq[1]))
        print('      -base: {:.3f}'.format(nne_base_seq[1]))
        print('    -rotation')
        print('      -Censi: {:.3f}'.format(nne_censi_seq[0]))
        print('      -Monte-Carlo 65: {:.3f}'.format(nne_mc_65_seq[0]))
        print('      -proposed: {:.3f}'.format(nne_prop_seq[0]))
        print('      -base: {:.3f}'.format(nne_base_seq[0]))

    kl_div_censi /= n_tot
    kl_div_mc_65 /= n_tot
    kl_div_prop /= n_tot
    kl_div_base /= n_tot

    nne_censi /= n_tot
    nne_mc_65 /= n_tot
    nne_prop /= n_tot
    nne_base /= n_tot

    nne_censi = np.sqrt(nne_censi)
    nne_mc_65 = np.sqrt(nne_mc_65)
    nne_prop = np.sqrt(nne_prop)
    nne_base = np.sqrt(nne_prop)

    # display results
    print('Covariance results')
    print('  Kullback-Leibler divergence')
    print('    -translation')
    print('      -Censi: {:.3f}'.format(kl_div_censi[1]))
    print('      -Monte-Carlo 65: {:.3f}'.format(kl_div_mc_65[1]))
    print('      -proposed: {:.3f}'.format(kl_div_prop[1]))
    print('      -base: {:.3f}'.format(kl_div_base[1]))
    print('    -rotation')
    print('      -Censi: {:.3f}'.format(kl_div_censi[0]))
    print('      -Monte-Carlo 65: {:.3f}'.format(kl_div_mc_65[0]))
    print('      -proposed: {:.3f}'.format(kl_div_prop[0]))
    print('      -base: {:.3f}'.format(kl_div_base[0]))
    print('  Normalized Norm Error')
    print('    -translation')
    print('      -Censi: {:.3f}'.format(nne_censi[1]))
    print('      -Monte-Carlo 65: {:.3f}'.format(nne_mc_65[1]))
    print('      -proposed: {:.3f}'.format(nne_prop[1]))
    print('      -base: {:.3f}'.format(nne_base[1]))
    print('    -rotation')
    print('      -Censi: {:.3f}'.format(nne_censi[0]))
    print('      -Monte-Carlo 65: {:.3f}'.format(nne_mc_65[0]))
    print('      -proposed: {:.3f}'.format(nne_prop[0]))
    print('      -base: {:.3f}'.format(nne_base[0]))
