
from __init__ import *
from utils import *


def cov_plot(dataset, sequence, scan_ref, scan_in):
    base_path = os.path.join(Param.results_path, sequence, str(scan_ref))
    T_gt = dataset.get_data(sequence)
    T_init = SE3.mul(SE3.inv(T_gt[scan_ref]), T_gt[scan_in])
    T_mc, T_init_mc = dataset.get_mc_results(sequence, scan_ref)
    f_metrics = os.path.join(base_path, 'metrics.p')
    metrics = dataset.load(f_metrics)

    # compute and divide Monte-Carlo into inliers and outliers
    mc_new = np.zeros((Param.n_mc, 6))
    for n in range(Param.n_mc):
        T_diff = SE3.mul(T_mc[n], T_init)
        mc_new[n] = SE3.log(T_diff)  #  xi = log( T_hat * T^{-1} )

    T_ut, T_init_ut = dataset.get_ut_results(sequence, scan_ref)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlabel='x (m)', ylabel=' y (m)', title="Results projected on the ground plane for visualization")
    fig.suptitle(sequence + ", scan ref:" + str(scan_ref) + ", scan in: " + str(scan_in),
                 fontsize=16)

    ax.scatter(T_init_mc[:, 0, 3], T_init_mc[:, 1, 3], c='yellow')
    ax.scatter(T_mc[:, 0, 3], T_mc[:, 1, 3], c='tomato')

    ax.scatter(T_init_ut[:, 0, 3], T_init_ut[:, 1, 3], c='cyan')
    ax.scatter(T_ut[:, 0, 3], T_ut[:, 1, 3], c='blue')
    ax.scatter(T_ut[0, 0, 3], T_ut[0, 1, 3], c='red')

    ax.scatter(T_init[0, 3], T_init[1, 3], c='black')

    # compute and plot 3-sigma covariances
    cov_censi = metrics['cov_censi']
    cov_prop = metrics['cov_prop']
    cov_ut = metrics['cov_ut']
    cov_mc_65 = metrics['cov_mc_65']
    cov_mc = metrics['cov_mc']
    cov_mc_inliers = metrics['cov_mc_inliers']

    xy = contour_ellipse(T_ut[0], cov_censi, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='brown')

    xy = contour_ellipse(T_ut[0], cov_ut, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='red')

    xy = contour_ellipse(T_ut[0], cov_mc_65, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='orange')

    xy = contour_ellipse(T_ut[0], cov_mc_inliers, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='tomato')

    xy = contour_ellipse(T_ut[0], cov_mc, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='grey')

    xy = contour_ellipse(T_ut[0], cov_prop, sigma=3, alpha=0.2)
    plt.plot(xy[:, 0], xy[:, 1], c='blue')
    ax.legend([r'$\mathbf{\hat{Q}}_{\mathrm{censi}}$',
               r'$\mathbf{\hat{Q}}^{\mathrm{large}}_{\mathrm{scale}}$',
               r'$\mathbf{\hat{Q}}^{\mathrm{monte}}_{\mathrm{carlo}}$ (65 MC)',
               r'$\mathbf{\hat{Q}}^{\mathrm{monte}}_{\mathrm{carlo}}$ (1000 MC)',
               r'$\mathbf{\hat{Q}}_{\mathrm{icp}}$ (proposed)',
               r'$\mathbf{\hat{T}}_{\mathrm{odo}}$ (inliers)',
               r'$\mathbf{\hat{T}}_{\mathrm{odo}}$ (outliers)',
               r'$\mathbf{\hat{T}}_{\mathrm{icp}}$ (inliers)',
               r'$\mathbf{\hat{T}}_{\mathrm{icp}}$ (outliers)',
               r'$\mathbf{\hat{T}}_{\mathrm{odo}}$ (sigma points)',
               r'$\mathbf{\hat{T}}_{\mathrm{icp}}$ (sigma points)',
               r'$\mathbf{\hat{T}}_{\mathrm{icp}}$ (init at $\mathbf{T}$)',
               r'$\mathbf{T}$'
               ])
    ax.axis('equal')
    plt.show()
    fig.savefig(os.path.join(base_path,  "cov.png"))


def pg_plot(dataset, sequence):
    base_path = os.path.join(Param.results_path, sequence)
    T_gt = dataset.get_data(sequence)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlabel='x (m)', ylabel=' y (m)',
           title="Results projected on the ground plane for visualization")
    fig.suptitle(sequence.replace('_', ' '), fontsize=16)

    def get_results(base_path, sensor, n):
        file = os.path.join(base_path, 'pg_' + sensor + '_' + str(n) + '.p')
        mondict = dataset.load(file)
        return mondict['T'], mondict['cov']

    for n in range(Param.n_pg):
        T_odo, cov_odo = get_results(base_path, 'odo', n)
        T_icp, cov_icp = get_results(base_path, 'icp', n)
        T_odo_icp, cov_odo_icp = get_results(base_path, 'odo_icp', n)
        T_prop, cov_prop = get_results(base_path, 'prop', n)

        if n == 0:
            ax.plot(T_gt[:, 0, 3], T_gt[:, 1, 3], c='black')

        ax.plot(T_odo[:, 0, 3], T_odo[:, 1, 3], c='orange')
        ax.plot(T_icp[:, 0, 3], T_icp[:, 1, 3], c='red')
        ax.plot(T_odo_icp[:, 0, 3], T_odo_icp[:, 1, 3], c='blue')
        ax.plot(T_prop[:, 0, 3], T_prop[:, 1, 3], c='green')

    alpha = 0.9
    sigma = 3
    xy = contour_ellipse(T_odo[-1], cov_odo[-1], sigma=sigma, alpha=alpha)
    plt.plot(xy[:, 0], xy[:, 1], c='orange', linestyle='dashed')

    xy = contour_ellipse(T_icp[-1], cov_icp[-1], sigma=sigma, alpha=alpha)
    plt.plot(xy[:, 0], xy[:, 1], c='red', linestyle='dashed')

    xy = contour_ellipse(T_odo_icp[-1], cov_odo_icp[-1], sigma=sigma, alpha=alpha)
    plt.plot(xy[:, 0], xy[:, 1], c='blue', linestyle='dashed')

    xy = contour_ellipse(T_prop[-1], cov_prop[-1], sigma=sigma, alpha=alpha)
    plt.plot(xy[:, 0], xy[:, 1], c='green', linestyle='dashed')

    ax.legend([r'$\mathbf{T}$',
               r'odo.',
               r'ICP',
               r'odo.+ICP',
               r'proposed',
               ])
    ax.axis('equal')

    plt.show()
    fig.savefig(os.path.join(base_path,  "odometry.png"))


def ellipse(T, cov, sigma=3):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)

    I = np.argsort(vals)[::-1]
    a = sigma*np.sqrt(vals[0])
    b = sigma*np.sqrt(vals[1])
    mmax = 50
    xy = np.zeros((mmax, 2))
    for m in range(mmax):
        v = -np.pi + 2*np.pi*m/(mmax-1)
        p = a*vecs[:, I[0]]*np.sin(v) + b*vecs[:, I[1]]*np.cos(v)

        r = p[:2]
        xy[m, 0] = r[0]
        xy[m, 1] = r[1]
    xy += T[:2, 2]
    return xy


def contour_ellipse(T, cov, sigma=2, alpha=0., mmax=50):
    xy = ellipse_se3(T, cov, sigma, mmax)
    alpha_shape = alphashape.alphashape(xy, alpha)
    a = alpha_shape.exterior.coords.xy[0]
    b = alpha_shape.exterior.coords.xy[1]
    xy_con = np.zeros((len(a), 2))
    xy_con[:, 0] = a
    xy_con[:, 1] = b
    return xy_con


def ellipse_se3(T, cov, sigma=2, mmax=50):
    # remove z axis as unnused
    cov[5] = 0
    cov[:, 5] = 0

    # cov[:2] = 0
    # cov[:, :2] = 0

    def eigsorted(cov):
        vals, vecs = np.linalg.eig(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)

    a = sigma*np.sqrt(vals[0])
    b = sigma*np.sqrt(vals[1])
    c = sigma*np.sqrt(vals[2])

    va = vecs[:, 0]
    vb = vecs[:, 1]
    vc = vecs[:, 2]

    xy = np.zeros((mmax**2, 2))
    for m in range(mmax):
        for n in range(mmax):
            v = - np.pi + 2*np.pi*m/(mmax-1)
            u = - np.pi + 2*np.pi*n/(mmax-1)
            xi = a*va*np.sin(v)*np.cos(u) + b*vb*np.cos(v)*np.cos(u) + c*vc*np.sin(u)
            r = SE3.exp(xi).dot(T)[:2, 3]
            xy[m+n*mmax] = r
    return xy