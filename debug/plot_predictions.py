#  M = scores_spocreg.mean(axis=2)
#  opt = np.unravel_index(M.argmin(), M.shape)
#  err_spocreg = scores_spocreg[opt]
#  err_nospoc = scores_nospoc
#  ax2.scatter(err_nospoc, err_spocreg)
#  lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),
#          np.max([ax2.get_xlim(), ax2.get_ylim()])]
#  ax2.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
#  ax2.set_aspect('equal')
#  ax2.set_xlim(lims)
#  ax2.set_ylim(lims)
#
#  ax2.set_xlabel('No SPoC')
#  ax2.set_ylabel('Best SPoC reg')
#  ax2.set_title('Prediction errors accross CV folds\n'
#                '(Best = %d compo, %.4f alpha)'
#                % (components[opt[0]], alpha[opt[1]]))
#
#  plt.tight_layout()
#  plt.show()
#  fig.savefig(op.join(cfg.path_outputs, 'plot_MAE_allch.png'), dpi=300)
#
#  # Predictions scatter plots
#  predictions = np.load(op.join(cfg.path_outputs,
#                        'all_predict_mag_shrinktrace.npy')).item()
#  AE_nospoc = predictions['nospoc']
#  AE_bestspocreg = predictions['bestspocreg']
#  CVscores_nospoc = predictions['scores_nospoc']
#  CVscores_bestspocreg = predictions['scores_bestspocreg']
#
#  plt.close('all')
#
#  # scatter plot
#  plt.figure(figsize=(8, 7))
#  plt.scatter(AE_nospoc, AE_bestspocreg, alpha=0.3)
#
#  plt.xlim(-1, 50)
#  plt.xlabel('NoSPoC')
#  plt.xticks([1, 5, 10, 20, 30, 40, 50], [1, 5, 10, 20, 30, 40, 50])
#
#  plt.ylim(-1, 50)
#  plt.ylabel('Best SPoC reg')
#  plt.yticks([1, 5, 10, 20, 30, 40, 50], [1, 5, 10, 20, 30, 40, 50])
#  plt.plot((0, 50), (0, 50), 'k-', linestyle='--', zorder=1000)
#  plt.tight_layout()
#  plt.savefig(op.join(cfg.path_outputs, 'plot_AE_samples.png'), dpi=300)
#
#  # log scatter plot
#  plt.figure(figsize=(8, 7))
#  plt.scatter(np.log10(AE_nospoc), np.log10(AE_bestspocreg), alpha=0.3)
#
#  plt.xlim(-1, np.log10(50))
#  plt.xlabel('NoSPoC')
#  plt.xticks(np.log10([1, 5, 10, 20, 30, 40, 50]),
#             [1, 5, 10, 20, 30, 40, 50])
#
#  plt.ylim(-1, np.log10(50))
#  plt.ylabel('Best SPoC reg')
#  plt.yticks(np.log10([1, 5, 10, 20, 30, 40, 50]),
#             [1, 5, 10, 20, 30, 40, 50])
#  plt.plot((-1, 50), (-1, 50), 'k-', linestyle='--', zorder=1000)
#  plt.tight_layout()
#  plt.savefig(op.join(cfg.path_outputs, 'plot_AE_samples_log.png'), dpi=300)
#
#  # sanity check CV fold
#  plt.figure(figsize=(8, 7))
#  plt.scatter(CVscores_nospoc, CVscores_bestspocreg, alpha=0.3)
#
#  plt.xlim(8.4, 10.2)
#  plt.xlabel('NoSPoC')
#
#  plt.ylim(8.4, 10.2)
#  plt.ylabel('Best SPoC reg')
#
#  plt.plot((7.5, 10.5), (7.5, 10.5), 'k-', linestyle='--', zorder=1000)
#  plt.tight_layout()
#  plt.savefig(op.join(cfg.path_outputs, 'plot_AE_folds.png'), dpi=300)
#
#  # hexbin plot
#  plt.figure(figsize=(9, 7))
#  plt.hexbin(AE_nospoc, AE_bestspocreg,
#             gridsize=30, mincnt=1, cmap='viridis')
#  plt.xlim(0, 50)
#  plt.xlabel('NoSPoC')
#  plt.xticks([1, 5, 10, 20, 30, 40, 50],
#             [1, 5, 10, 20, 30, 40, 50])
#
#  plt.ylim(0, 50)
#  plt.ylabel('Best SPoC reg')
#  plt.yticks([1, 5, 10, 20, 30, 40, 50],
#             [1, 5, 10, 20, 30, 40, 50])
#  plt.plot((-1, 50), (-1, 50), 'r-', linestyle='--', zorder=1000)
#  plt.colorbar()
#  plt.tight_layout()
#
#  # log hexbin plot
#  plt.figure(figsize=(9, 7))
#  plt.hexbin(np.log10(AE_nospoc), np.log10(AE_bestspocreg),
#             gridsize=30, mincnt=1, cmap='viridis')
#  plt.xlim(-1, np.log10(50))
#  plt.xlabel('NoSPoC')
#  plt.xticks(np.log10([1, 5, 10, 20, 30, 40, 50]),
#             [1, 5, 10, 20, 30, 40, 50])
#
#  plt.ylim(-1, np.log10(50))
#  plt.ylabel('Best SPoC reg')
#  plt.yticks(np.log10([1, 5, 10, 20, 30, 40, 50]),
#             [1, 5, 10, 20, 30, 40, 50])
#  plt.plot((-1, 50), (-1, 50), 'r-', linestyle='--', zorder=1000)
#  plt.colorbar()
#  plt.tight_layout()
#  plt.savefig(op.join(cfg.path_outputs, 'plot_AE_density.png'), dpi=300)
