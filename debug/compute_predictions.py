# # Predictions
#
# predict_nospoc = cross_val_predict(ridge_nospoc, X=feat, y=y, cv=cv,
#                                    n_jobs=1)
# AE_nospoc = np.abs(predict_nospoc - y)
#
# bestspocreg = SPoC(covs=covs, fbands=cfg.fbands, spoc=True,
#                    n_components=70, alpha=0.5556)
# ridge_bestspocreg = make_pipeline(bestspocreg, StandardScaler(),
#                                   RidgeCV(alphas=np.logspace(-3, 5, 100)))
# predict_bestspocreg = cross_val_predict(ridge_bestspocreg, X, y, cv=cv,
#                                         n_jobs=1)
# AE_bestspocreg = np.abs(predict_bestspocreg - y)
#
# # for sanity check vs above
# CVscores_nospoc = [np.abs(predict_nospoc[ff[1]] - y[ff[1]]).mean()
#                    for ff in cv.split(X)]
# CVscores_bestspocreg = [np.abs(predict_bestspocreg[ff[1]] - y[ff[1]]).mean()
#                         for ff in cv.split(X)]
#
# predictions = {'nospoc': np.array(AE_nospoc),
#                'bestspocreg': np.array(AE_bestspocreg),
#                'scores_nospoc': np.array(CVscores_nospoc),
#                'scores_bestspocreg': np.array(CVscores_bestspocreg)}
#
# np.save(op.join(cfg.path_outputs, 'all_predict_mag_shrinktrace.npy'),
#         predictions)

# spoc = SPoC(covs=covs, fbands=cfg.fbands, spoc=True, n_components=len(picks))
# spoc.fit(X, y)
# plt.matshow(spoc.filters_[4], cmap='RdBu')
# features = spoc.transform(X)
# plt.figure()
# plt.plot(features[4])
# plt.figure()
# plt.plot(features.reshape(len(y), len(picks), len(cfg.fbands))[4])
# plt.figure(figsize=(8,6))
# for ii, comp in enumerate(features[4].reshape(102,9)):
#     plt.plot(comp)
