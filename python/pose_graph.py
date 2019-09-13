from __init__ import *
from utils import *
import plots


def compute(dataset, sequence):
    # prepare data file for the pose graph
    T_gt = dataset.get_data(sequence)

    T_odo = np.zeros((Param.n_pg, T_gt.shape[0]-1, 4, 4))
    T_icp = np.zeros((Param.n_pg, T_odo[0].shape[0], 4, 4))

    cov_odo = np.zeros((Param.n_pg, T_odo[0].shape[0], 6, 6))
    cov_prop = np.zeros((Param.n_pg, T_odo[0].shape[0], 6, 6))
    cov_full = np.zeros((Param.n_pg, T_odo[0].shape[0], 12, 12))

    def subroutine(base_path, sensor, n, T_1, cov_1, T_2=None, cov_2=None, b_cross=False):
        f_name = os.path.join(base_path, 'pg_results_' + sensor + '_' + str(n) + '.txt')
        cov_f_name = os.path.join(base_path, 'pg_cov_' + sensor + '_' + str(n) + '.txt')
        out_f_name = os.path.join(base_path, 'pg_' + sensor + '_' + str(n) + '.p')

        if T_2 is None:
            T, cov = odo_one_sensor(T_1[n], cov_1[n])
        elif not b_cross:
            T, cov = odo_without_cross(T_1[n], T_2[n], cov_1[n], cov_2[n])
        else:
            T, cov = odo_with_cross(T_1[n], T_2[n], cov_1[n])
        write_as_gtsam(T, cov, f_name, cov_f_name)
        gtsam_to_python(dataset, f_name, cov_f_name, out_f_name)

    for nn in range(T_odo[0].shape[0]):
        base_path = os.path.join(Param.results_path, sequence, str(nn))
        f_metrics = os.path.join(base_path, 'pg_metrics.p')
        metrics = dataset.load(f_metrics)
        T_odo[:, nn] = metrics['T_odo']
        T_icp[:, nn] = metrics['T_icp']
        cov_prop[:, nn] = metrics['cov_prop']
        cov_full[:, nn, 6:, 6:] = metrics['cov_prop']
        cov_full[:, nn, :6, 6:] = metrics['cov_cross']

        for n in range(Param.n_pg):
            cov_odo[n, nn] = Param.pg_Q_odo
            cov_full[n, nn, :6, :6] = Param.pg_Q_odo

    cov_full[:, :, 6:, :6] = np.transpose(cov_full[:, :, :6, 6:], (0, 1, 3, 2))

    base_path = os.path.join(Param.results_path, sequence)
    for n in range(Param.n_pg):
        subroutine(base_path, 'odo', n, T_odo, cov_odo)
        subroutine(base_path, 'icp', n, T_icp, cov_prop)
        subroutine(base_path, 'odo_icp', n, T_odo, cov_odo, T_icp, cov_prop)
        subroutine(base_path, 'prop', n, T_odo, cov_full, T_icp, b_cross=True)


def odo_one_sensor(T_inc, cov_inc):
    T = SE3.new(T_inc.shape[0]+1)
    cov = SE3.new_cov(T_inc.shape[0]+1)

    for n in range(1, T.shape[0]):
        T[n] = SE3.normalize(SE3.mul(T[n-1], T_inc[n-1]))
        Ad = SE3.Ad(T[n-1])
        cov[n] = cov[n-1] + Ad.dot(cov_inc[n-1]).dot(Ad.T)
    return T, cov


def fuse_without_cross(T_1, T_2, cov_1, cov_2):
    T_est = T_1
    inv_cov_1 = np.linalg.inv(cov_1)
    inv_cov_2 = np.linalg.inv(cov_2)
    inv_T_1 = SE3.inv(T_1)
    inv_T_2 = SE3.inv(T_2)
    for i in range(10):  #  Gauss-Newton iterations
        xi_1 = SE3.log(SE3.mul(T_est, inv_T_1))
        xi_2 = SE3.log(SE3.mul(T_est, inv_T_2))
        invJ_1 = SE3.jacInv(xi_1)
        invJ_2 = SE3.jacInv(xi_2)
        invJt1 = invJ_1.T.dot(inv_cov_1)
        invJt2 = invJ_2.T.dot(inv_cov_2)
        LHS = invJt1.dot(invJ_1) + invJt2.dot(invJ_2)
        RHS = invJt1.dot(xi_1) + invJt2.dot(xi_2)

        xi = -np.linalg.solve(LHS, RHS)
        T_est = SE3.mul(SE3.exp(xi), T_est)

    xi_1 = SE3.log(SE3.mul(T_est, inv_T_1))
    xi_2 = SE3.log(SE3.mul(T_est, inv_T_2))
    invJ_1 = SE3.jacInv(xi_1)
    invJ_2 = SE3.jacInv(xi_2)
    cov_est = np.linalg.inv(invJ_1.T.dot(inv_cov_1).dot(invJ_1) +
                            invJ_2.T.dot(inv_cov_2).dot(invJ_2))
    return T_est, cov_est


def fuse_with_cross(T_1, T_2, cov):
    A = np.zeros((12, 6))
    A[:6] = np.eye(6)
    A[6:] = np.eye(6)
    T_est = T_2
    cov_est = cov[6:, 6:]
    inv_cov = np.linalg.inv(cov)
    inv_T_1 = SE3.inv(T_1)
    inv_T_2 = SE3.inv(T_2)
    for i in range(10):  #  Gauss-Newton iterations
        xi_1 = SE3.log(SE3.mul(T_est, inv_T_1))
        xi_2 = SE3.log(SE3.mul(T_est, inv_T_2))
        xi = np.hstack((xi_1, xi_2))
        invJ_1 = SE3.jacInv(xi_1)
        invJ_2 = SE3.jacInv(xi_2)

        invJ = np.eye(12)
        invJ[:6, :6] = invJ_1
        invJ[6:, 6:] = invJ_2
        invJt = invJ.T.dot(inv_cov).dot(invJ)

        LHS = A.T.dot(invJt.dot(invJ)).dot(A)
        RHS = A.T.dot(invJt.dot(xi))

        xi = -np.linalg.solve(LHS, RHS)
        T_est = SE3.mul(SE3.exp(xi), T_est)

        xi_1 = SE3.log(SE3.mul(T_est, SE3.inv(T_1)))
        xi_2 = SE3.log(SE3.mul(T_est, SE3.inv(T_2)))
        invJ_1 = SE3.jacInv(xi_1)
        invJ_2 = SE3.jacInv(xi_2)
        invJ = np.eye(12)
        invJ[:6, :6] = invJ_1
        invJ[6:, 6:] = invJ_2
        JinfoJ = (invJ.dot(A)).T.dot(inv_cov).dot(invJ.dot(A))
        cov_est = np.linalg.inv(JinfoJ)

    return T_est, cov_est


def odo_without_cross(T_1, T_2, cov_1, cov_2):
    T = SE3.new(T_2.shape[0]+1)
    cov = SE3.new_cov(T_1.shape[0]+1)
    # perfect initialization at identity
    for n in range(1, T.shape[0]):
        DeltaT_n, Delta_cov_n = fuse_without_cross(T_1[n-1], T_2[n-1], cov_1[n-1], cov_2[n-1])
        T[n] = SE3.normalize(SE3.mul(T[n-1], DeltaT_n))
        Ad = SE3.Ad(T[n-1])
        cov[n] = cov[n-1] + Ad.dot(Delta_cov_n).dot(Ad.T)
    return T, cov


def odo_with_cross(T_1, T_2, cov_inc):
    T = SE3.new(T_1.shape[0]+1)
    cov = SE3.new_cov(T_1.shape[0]+1)
    # perfect initialization at identity
    for n in range(1, T.shape[0]):
        DeltaT_n, Delta_cov_n = fuse_with_cross(T_1[n-1], T_2[n-1], cov_inc[n-1])
        xi = SE3.log(SE3.mul(T_1[n-1], SE3.inv(T_2[n-1])))
        xi_rot = np.linalg.norm(xi[:3])/3
        xi_t = np.linalg.norm(xi[3:])/3
        if xi_rot > Param.pg_std_rot*np.sqrt(3) or xi_t > Param.pg_std_pos*np.sqrt(3):
            DeltaT_n = T_1[n-1]
            Delta_cov_n = cov_inc[n-1][:6, :6]
        T[n] = SE3.normalize(SE3.mul(T[n-1], DeltaT_n))
        Ad = SE3.Ad(T[n-1])
        cov[n] = cov[n-1] + Ad.dot(Delta_cov_n).dot(Ad.T)
    return T, cov


def gtsam_to_python(dataset, f_name_results, f_name_cov, f_name_out):
    if not Param.b_pg_results and os.path.exists(f_name_out):
        mondict = dataset.load(f_name_out)
        T = mondict['T']
        cov = mondict['cov']
        return T, cov
    else:
        # get results
        T = np.zeros((0, 4, 4))
        file = open(f_name_results)
        line = file.readline().split()
        while line:
            if not line[0] == 'VERTEX_SE3:QUAT':
                break
            T_i = np.zeros((1, 4, 4))
            T_i[0, 3, 3] = 1
            T_i[0, :3, 3] = np.array(line[2:5]).astype(float)
            quat = np.array(line[5:10]).astype(float)
            if np.isnan(quat).sum():
                print(f_name_results)
                print(T.shape[0])
            quat = quat/np.linalg.norm(quat)
            T_i[0, :3, :3] = SO3.from_quaternion(quat, ordering='xyzw')
            T = np.concatenate((T, T_i))
            line = file.readline().split()
        file.close()

        # get cov
        data = np.genfromtxt(f_name_cov)
        cov = np.zeros((T.shape[0], 6, 6))
        for i in range(cov.shape[0]):
            cov[i] = data[i*6: (i+1)*6]
        mondict = {
            'T': T,
            'cov': cov
            }
        dataset.dump(mondict, f_name_out)
    return T, cov


def aggregate_results(dataset):
    def mse_mal_dist_sequence(T_true, T_hat, cov):
        i = T_true.shape[0]-1
        mse = np.zeros(2)
        xi = SE3.log(SE3.mul(SE3.inv(T_true[i]), T_hat[i]))
        mal_dist = SE3.mal_dist_rot_trans(xi, np.zeros(6), cov[i], T_hat[i])
        mse[0] = np.sum(np.abs(xi[:3])**2)
        mse[1] = np.sum(np.abs(xi[3:])**2)
        return mse, mal_dist

    def get_results(base_path, sensor, T_gt, n):
        f_results = os.path.join(base_path, 'pg_results_' + sensor + '_' + str(n) + '.txt')
        f_cov = os.path.join(base_path, 'pg_cov_' + sensor + '_' + str(n) + '.txt')
        f_out = os.path.join(base_path, 'pg_' + sensor + '_' + str(n) + '.p')
        T, cov = gtsam_to_python(dataset, f_results, f_cov, f_out)
        mse, mal_dist = mse_mal_dist_sequence(T_gt, T, cov)
        return mse, mal_dist

    full_mse_odo = np.zeros(2)
    full_mse_icp = np.zeros(2)
    full_mse_odo_icp = np.zeros(2)
    full_mse_prop = np.zeros(2)

    full_mal_dist_odo = np.zeros(2)
    full_mal_dist_icp = np.zeros(2)
    full_mal_dist_odo_icp = np.zeros(2)
    full_mal_dist_prop = np.zeros(2)

    n_tot = 0

    for sequence in dataset.sequences:
        base_path = os.path.join(Param.results_path, sequence)
        f_out = os.path.join(base_path, 'pg_metrics.p')
        if not Param.b_pg_results and os.path.exists(f_out):
            return
        T_gt = dataset.get_data(sequence)
    
        mse_odo = np.zeros((Param.n_pg, 2))
        mse_icp = np.zeros((Param.n_pg, 2))
        mse_odo_icp = np.zeros((Param.n_pg, 2))
        mse_prop = np.zeros((Param.n_pg, 2))
    
        mal_dist_odo = np.zeros((Param.n_pg, 2))
        mal_dist_icp = np.zeros((Param.n_pg, 2))
        mal_dist_odo_icp = np.zeros((Param.n_pg, 2))
        mal_dist_prop = np.zeros((Param.n_pg, 2))

        for n in range(Param.n_pg):
            mse, mal_dist = get_results(base_path, 'odo', T_gt, n)
            mal_dist_odo[n] = mal_dist
            mse_odo[n] = mse

            mse, mal_dist = get_results(base_path, 'icp', T_gt, n)
            mal_dist_icp[n] = mal_dist
            mse_icp[n] = mse

            mse, mal_dist = get_results(base_path, 'odo_icp', T_gt, n)
            mal_dist_odo_icp[n] = mal_dist
            mse_odo_icp[n] = mse

            mse, mal_dist = get_results(base_path, 'prop', T_gt, n)
            mal_dist_prop[n] = mal_dist
            mse_prop[n] = mse

        seuil_up = int(0.9*Param.n_pg)
        seuil_low = Param.n_pg - seuil_up

        mse_odo = np.sort(mse_odo.T).T[seuil_low: seuil_up]
        mse_icp = np.sort(mse_icp.T).T[seuil_low: seuil_up]
        mse_odo_icp = np.sort(mse_odo_icp.T).T[seuil_low: seuil_up]
        mse_prop = np.sort(mse_prop.T).T[seuil_low: seuil_up]

        mal_dist_odo = np.sort(mal_dist_odo.T).T[seuil_low: seuil_up]
        mal_dist_icp = np.sort(mal_dist_icp.T).T[seuil_low: seuil_up]
        mal_dist_odo_icp = np.sort(mal_dist_odo_icp.T).T[seuil_low: seuil_up]
        mal_dist_prop = np.sort(mal_dist_prop.T).T[seuil_low: seuil_up]

        mse_odo = mse_odo.sum(axis=0)
        mse_icp = mse_icp.sum(axis=0)
        mse_odo_icp = mse_odo_icp.sum(axis=0)
        mse_prop = mse_prop.sum(axis=0)

        mal_dist_odo = mal_dist_odo.sum(axis=0)
        mal_dist_icp = mal_dist_icp.sum(axis=0)
        mal_dist_odo_icp = mal_dist_odo_icp.sum(axis=0)
        mal_dist_prop = mal_dist_prop.sum(axis=0)

        full_mal_dist_odo += mal_dist_odo
        full_mse_odo += mse_odo
        full_mal_dist_icp += mal_dist_icp
        full_mse_icp += mse_icp

        full_mal_dist_odo_icp += mal_dist_odo_icp
        full_mse_odo_icp += mse_odo_icp
        full_mal_dist_prop += mal_dist_prop
        full_mse_prop += mse_prop

        n_tot += seuil_up-seuil_low
        tmp = 3*(seuil_up-seuil_low)
        mal_dist_odo = np.sqrt(mal_dist_odo/tmp)
        mal_dist_icp = np.sqrt(mal_dist_icp/tmp)
        mal_dist_odo_icp = np.sqrt(mal_dist_odo_icp/tmp)
        mal_dist_prop = np.sqrt(mal_dist_prop/tmp)

        mse_odo /= tmp
        mse_icp /= tmp
        mse_odo_icp /= tmp
        mse_prop /= tmp

        rmse_odo = np.sqrt(mse_odo**2)
        rmse_icp = np.sqrt(mse_icp**2)
        rmse_odo_icp = np.sqrt(mse_odo_icp**2)
        rmse_prop = np.sqrt(mse_prop**2)

        # display results
        print('Pose-graph results for sequence ' + sequence)
        print('  RMSE')
        print('    -translation (m)')
        print('      -odo.: {:.3f}'.format(rmse_odo[1]))
        print('      -ICP: {:.3f}'.format(rmse_icp[1]))
        print('      -odo.+ICP: {:.3f}'.format(rmse_odo_icp[1]))
        print('      -proposed: {:.3f}'.format(rmse_prop[1]))
        print('    -rotation (deg)')
        print('      -odo.: {:.2f}'.format(rmse_odo[0]*180/np.pi))
        print('      -ICP: {:.2f}'.format(rmse_icp[0]*180/np.pi))
        print('      -odo.+ICP: {:.2f}'.format(rmse_odo_icp[0]*180/np.pi))
        print('      -proposed: {:.2f}'.format(rmse_prop[0]*180/np.pi))
        print('  Mahalanobis distance')
        print('    -translation')
        print('      -odo.: {:.2f}'.format(mal_dist_odo[1]))
        print('      -ICP: {:.2f}'.format(mal_dist_icp[1]))
        print('      -odo.+ICP: {:.2f}'.format(mal_dist_odo_icp[1]))
        print('      -proposed: {:.2f}'.format(mal_dist_prop[1]))
        print('    -rotation')
        print('      -odo.: {:.2f}'.format(mal_dist_odo[0]))
        print('      -ICP: {:.2f}'.format(mal_dist_icp[0]))
        print('      -odo.+ICP: {:.2f}'.format(mal_dist_odo_icp[0]))
        print('      -proposed: {:.2f}'.format(mal_dist_prop[0]))

        mondict = {
            'rmse_odo': rmse_odo,
            'rmse_icp': rmse_icp,
            'rmse_odo_icp': rmse_odo_icp,
            'rmse_prop': rmse_prop,
            'mal_dist_odo': mal_dist_odo,
            'mal_dist_icp': mal_dist_icp,
            'mal_dist_odo_icp': mal_dist_odo_icp,
            'mal_dist_prop': mal_dist_prop,
        }
        dataset.dump(mondict, f_out)

    tmp = 3*n_tot
    full_mal_dist_odo = np.sqrt(full_mal_dist_odo/tmp)
    full_mal_dist_icp = np.sqrt(full_mal_dist_icp/tmp)
    full_mal_dist_odo_icp = np.sqrt(full_mal_dist_odo_icp/tmp)
    full_mal_dist_prop = np.sqrt(full_mal_dist_prop/tmp)

    full_rmse_odo = np.sqrt(full_mse_odo/tmp)
    full_rmse_icp = np.sqrt(full_mse_icp/tmp)
    full_rmse_odo_icp = np.sqrt(full_mse_odo_icp/tmp)
    full_rmse_prop = np.sqrt(full_mse_prop/tmp)

    print('Full results')
    print('  RMSE')
    print('    -translation (m)')
    print('      -odo.: {:.3f}'.format(full_rmse_odo[1]))
    print('      -ICP: {:.3f}'.format(full_rmse_icp[1]))
    print('      -odo.+ICP: {:.3f}'.format(full_rmse_odo_icp[1]))
    print('      -proposed: {:.3f}'.format(full_rmse_prop[1]))
    print('    -rotation (deg)')
    print('      -odo.: {:.2f}'.format(full_rmse_odo[0]*180/np.pi))
    print('      -ICP: {:.2f}'.format(full_rmse_icp[0]*180/np.pi))
    print('      -odo.+ICP: {:.2f}'.format(full_rmse_odo_icp[0]*180/np.pi))
    print('      -proposed: {:.2f}'.format(full_rmse_prop[0]*180/np.pi))
    print('  Mahalanobis distance')
    print('    -translation')
    print('      -odo.: {:.2f}'.format(full_mal_dist_odo[1]))
    print('      -ICP: {:.2f}'.format(full_mal_dist_icp[1]))
    print('      -odo.+ICP: {:.2f}'.format(full_mal_dist_odo_icp[1]))
    print('      -proposed: {:.2f}'.format(full_mal_dist_prop[1]))
    print('    -rotation')
    print('      -odo.: {:.2f}'.format(full_mal_dist_odo[0]))
    print('      -ICP: {:.2f}'.format(full_mal_dist_icp[0]))
    print('      -odo.+ICP: {:.2f}'.format(full_mal_dist_odo_icp[0]))
    print('      -proposed: {:.2f}'.format(full_mal_dist_prop[0]))


def results_latex(dataset, sequence):
    base_path = os.path.join(Param.results_path, sequence)
    T_gt = dataset.get_data(sequence)

    f_name = os.path.join(Param.latex_path, sequence + 'T.txt')
    header = "x_true y_true"
    data = np.zeros((T_gt.shape[0], 2))
    data[:, 0] = T_gt[:, 0, 3]
    data[:, 1] = T_gt[:, 1, 3]
    np.savetxt(f_name, data, comments='', header=header)

    header = "x_odo y_odo x_icp y_icp x_odo_icp y_odo_icp x_prop y_prop"

    def get_results(base_path, sensor, n):
        f_results = os.path.join(base_path, 'pg_results_' + sensor + '_' + str(n) + '.txt')
        f_cov = os.path.join(base_path, 'pg_cov_' + sensor + '_' + str(n) + '.txt')
        f_out = os.path.join(base_path, 'pg_' + sensor + '_' + str(n) + '.p')
        T, cov = gtsam_to_python(dataset, f_results, f_cov, f_out)
        return T, cov

    for n in range(Param.n_pg):
        T_odo, cov_odo = get_results(base_path, 'odo', n)
        T_icp, cov_icp = get_results(base_path, 'icp', n)
        T_odo_icp, cov_odo_icp = get_results(base_path, 'odo_icp', n)
        T_prop, cov_prop = get_results(base_path, 'prop', n)

        f_name = os.path.join(Param.latex_path, sequence + str(n) + 'pgT.txt')
        data = np.zeros((T_odo.shape[0], 8))
        data[:, :2] = T_odo[:, :2, 3]
        data[:, 2:4] = T_icp[:, :2, 3]
        data[:, 4:6] = T_odo_icp[:, :2, 3]
        data[:, 6:8] = T_prop[:, :2, 3]
        np.savetxt(f_name, data, comments='', header=header)

    f_name = os.path.join(Param.latex_path, sequence + 'Qfinal.txt')
    header = "x_odo y_odo x_icp y_icp x_odo_icp y_odo_icp x_prop y_prop"

    alpha = 0.4
    sigma = 3
    xy_odo = plots.contour_ellipse(T_odo[-1], cov_odo[-1], sigma=sigma, alpha=alpha)
    xy_icp = plots.contour_ellipse(T_icp[-1], cov_icp[-1], sigma=sigma, alpha=alpha)
    xy_odo_icp = plots.contour_ellipse(T_odo_icp[-1], cov_odo_icp[-1], sigma=2, alpha=alpha)
    xy_prop = plots.contour_ellipse(T_prop[-1], cov_prop[-1], sigma=sigma, alpha=alpha)

    n_min = np.min([150, xy_odo.shape[0], xy_prop.shape[0], xy_icp.shape[0], xy_odo_icp.shape[0]])
    xy_odo = xy_odo[np.linspace(0, xy_odo.shape[0]-1, n_min, dtype=int)]
    xy_prop = xy_prop[np.linspace(0, xy_prop.shape[0]-1, n_min, dtype=int)]
    xy_icp = xy_icp[np.linspace(0, xy_icp.shape[0]-1, n_min, dtype=int)]
    xy_odo_icp = xy_odo_icp[np.linspace(0, xy_odo_icp.shape[0]-1, n_min, dtype=int)]

    data = np.zeros((xy_odo.shape[0], 8))
    data[:, :2] = xy_odo
    data[:, 2:4] = xy_icp
    data[:, 4:6] = xy_odo_icp
    data[:, 6:8] = xy_prop
    np.savetxt(f_name, data, comments='', header=header)


def pg_icp(dataset, sequence):
    T_gt = dataset.get_data(sequence)
    base_path = os.path.join(Param.results_path, sequence, str(T_gt.shape[0]-1))
    path = os.path.join(base_path, "pg_ut_12_" + str(Param.n_pg-2) + ".txt")
    if not Param.b_pg_icp and os.path.exists(path):
        print(path + " already exist")
        for scan_ref in range(T_gt.shape[0]-1):
            results(dataset, sequence, scan_ref)
        return

    for n in range(Param.n_pg):
        for scan_ref in range(0, T_gt.shape[0]-1):
            base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
            scan_in = scan_ref + 1
            pc_ref = dataset.get_pc(sequence, scan_ref)
            pc_in = dataset.get_pc(sequence, scan_in)
            T_true = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])

            # sample initial transformation
            xi = np.hstack((np.random.normal(0, Param.pg_std_rot, 3),
                            np.random.normal(0, Param.pg_std_pos, 3)))
            T_odo = SE3.normalize(SE3.mul(SE3.exp(-xi), T_true))  # T = exp(xi) T_hat

            pose_path = os.path.join(base_path, "pg_T_censi" + str(n) + ".txt")
            cov_path = os.path.join(base_path, "pg_cov_censi_" + str(n) + ".txt")
            if Param.b_pg_icp or not os.path.exists(cov_path):
                icp_with_cov(pc_ref, pc_in, T_odo, Param.config_yaml, pose_path, cov_path)

            # sigma-points
            sps = Param.pg_ut.sp.sigma_points(Param.pg_Q_odo)
            file_sp = os.path.join(base_path, "pg_sp_" + str(n) + ".txt")
            np.savetxt(file_sp, sps)
            # unscented transform
            for nn in range(13):
                path = os.path.join(base_path, "pg_ut_" + str(nn) + "_" + str(n) + ".txt")
                if not Param.b_pg_icp and os.path.exists(path):
                    print(path + " already exist")
                    continue

                T_init_n = SE3.mul(SE3.exp(sps[nn]), T_odo)  #  T_sp = exp(xi) T_hat
                T_init_n = SE3.normalize(T_init_n)
                icp_without_cov(pc_ref, pc_in, T_init_n, path)

    for scan_ref in range(T_gt.shape[0]-1):
        results(dataset, sequence, scan_ref)


def results(dataset, sequence, scan_ref):
    """Get ICP results for a trajectory"""
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    f_metrics = os.path.join(base_path, 'pg_metrics.p')
    if not Param.b_pg_results and os.path.exists(f_metrics):
        print(f_metrics + " already exists")
        return

    cov_cross = SE3.new_cov(Param.n_pg)
    cov_prop = SE3.new_cov(Param.n_pg)
    T_odo = SE3.new(Param.n_pg)
    T_icp = SE3.new(Param.n_pg)

    for n in range(Param.n_pg):
        cov_path = os.path.join(base_path, "pg_cov_censi_" + str(n) + ".txt")
        data = np.genfromtxt(cov_path)
        T_ut, T_init_ut = dataset.get_pg_ut_results(sequence, scan_ref, n)

        cov_sc = Param.std_sensor**2 * data[6:]
        _, _, cov_ut, cov_cross_n = Param.pg_ut.unscented_transform_se3(T_ut)
        cov_prop[n] = cov_sc + cov_ut
        cov_cross[n] = cov_cross_n
        T_odo[n] = T_init_ut[0]
        T_icp[n] = T_ut[0]

    metrics = {
        'cov_prop': cov_prop,
        'cov_cross': cov_cross,
        'T_odo': T_odo,
        'T_icp': T_icp,
        }
    dataset.dump(metrics, f_metrics)
