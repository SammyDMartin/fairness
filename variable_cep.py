from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from aif360.metrics.classification_metric import ClassificationMetric

class VariableCEP(CalibratedEqOddsPostprocessing):
    def set_NP(self, NP_rate):
        self.fn_rate = NP_rate[0]
        self.fp_rate = NP_rate[1]
    

def normed_rates(fp_rate, fn_rate):
    norm_const = float(fp_rate + fn_rate) if\
                      (fp_rate != 0 and fn_rate != 0) else 1
    return (fp_rate / norm_const), (fn_rate / norm_const)


from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification

import numpy as np
class InquisitiveRejectOptionClassification(RejectOptionClassification):

  def fit(self, dataset_true, dataset_pred, metric_fn=None):
      """Estimates the optimal classification threshold and margin for reject
      option classification that optimizes the metric provided.
      Note:
          The `fit` function is a no-op for this algorithm.
      Args:
          dataset_true (BinaryLabelDataset): Dataset containing the true
              `labels`.
          dataset_pred (BinaryLabelDataset): Dataset containing the predicted
              `scores`.
      Returns:
          RejectOptionClassification: Returns self.
      """
      fair_metric_arr = np.zeros(self.num_class_thresh*self.num_ROC_margin)
      balanced_acc_arr = np.zeros_like(fair_metric_arr)
      ROC_margin_arr = np.zeros_like(fair_metric_arr)
      class_thresh_arr = np.zeros_like(fair_metric_arr)

      cnt = 0
      # Iterate through class thresholds
      for class_thresh in np.linspace(self.low_class_thresh,
                                      self.high_class_thresh,
                                      self.num_class_thresh):

          self.classification_threshold = class_thresh
          if class_thresh <= 0.5:
              low_ROC_margin = 0.0
              high_ROC_margin = class_thresh
          else:
              low_ROC_margin = 0.0
              high_ROC_margin = (1.0-class_thresh)

          # Iterate through ROC margins
          for ROC_margin in np.linspace(
                              low_ROC_margin,
                              high_ROC_margin,
                              self.num_ROC_margin):
              self.ROC_margin = ROC_margin

              # Predict using the current threshold and margin
              dataset_transf_pred = self.predict(dataset_pred)

              dataset_transf_metric_pred = BinaryLabelDatasetMetric(
                                            dataset_transf_pred,
                                            unprivileged_groups=self.unprivileged_groups,
                                            privileged_groups=self.privileged_groups)
              classified_transf_metric = ClassificationMetric(
                                            dataset_true,
                                            dataset_transf_pred,
                                            unprivileged_groups=self.unprivileged_groups,
                                            privileged_groups=self.privileged_groups)

              ROC_margin_arr[cnt] = self.ROC_margin
              class_thresh_arr[cnt] = self.classification_threshold

              # Balanced accuracy and fairness metric computations
              balanced_acc_arr[cnt] = 0.5*(classified_transf_metric.true_positive_rate()\
                                      +classified_transf_metric.true_negative_rate())
              
              ### THE ONLY CHANGE I MAKE TO THE FUNCTION IS HERE
              if metric_fn:
                  fair_metric_arr[cnt] = metric_fn(classified_transf_metric)
              ###
              
              elif self.metric_name == "Statistical parity difference":
                  fair_metric_arr[cnt] = dataset_transf_metric_pred.mean_difference()
              elif self.metric_name == "Average odds difference":
                  fair_metric_arr[cnt] = classified_transf_metric.average_odds_difference()
              elif self.metric_name == "Equal opportunity difference":
                  fair_metric_arr[cnt] = classified_transf_metric.equal_opportunity_difference()
              
              

              cnt += 1

      rel_inds = np.logical_and(fair_metric_arr >= self.metric_lb,
                                fair_metric_arr <= self.metric_ub)
      if any(rel_inds):
          best_ind = np.where(balanced_acc_arr[rel_inds]
                              == np.max(balanced_acc_arr[rel_inds]))[0][0]
      else:
          print("Unable to satisfy fairness constraints")
          rel_inds = np.ones(len(fair_metric_arr), dtype=bool)
          best_ind = np.where(fair_metric_arr[rel_inds]
                              == np.min(fair_metric_arr[rel_inds]))[0][0]

      self.ROC_margin = ROC_margin_arr[rel_inds][best_ind]
      self.classification_threshold = class_thresh_arr[rel_inds][best_ind]

      return self
