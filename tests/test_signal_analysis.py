import pytest

from vgx.gpqa.signal_analysis import cross_validate


def rows():
    return [{'item_id':str(i),'partition':'calibration','generator_answer':'A',
             'generator_p_correct':.99 if i%3 else .7,
             'correct_index':0 if i%3 else 1,'outcome':int(bool(i%3)),
             'verifiers':{'binary':{'p_correct':float(bool(i%3))},
                          'probability':{'p_correct':.9 if i%3 else .1}}} for i in range(30)]


def test_oof_calibration_and_binary_likelihoods_are_fitted_only_on_training_rows(monkeypatch):
    from vgx.gpqa.offline_validation import CorrectnessForecast
    original=CorrectnessForecast.fit;fit_sizes=[]
    def checked(self,train):
        assert all(r['partition']=='calibration' for r in train)
        fit_sizes.append(len(train))
        return original(self,train)
    monkeypatch.setattr(CorrectnessForecast,'fit',checked)
    result=cross_validate(rows(),['binary','probability'],{'binary':'binary','probability':'probability'})
    assert result['n']==30 and result['wrong']==10
    assert fit_sizes==[20]*9
    for fold in result['folds']:
        assert not set(fold['train'])&set(fold['validation'])
    assert sorted(i for fold in result['folds'] for i in fold['validation'])==sorted(r['item_id'] for r in rows())
    assert 'joint:binary' in result['metrics'] and 'bayes:probability' in result['metrics']
    assert all(0<p<1 for p in result['predictions']['bayes:binary'])


def test_too_few_errors_is_reported_without_claiming_a_fit():
    result=cross_validate(rows()[:3],[],{})
    assert result['status']=='insufficient_correctness_classes_for_folds'
    assert 'metrics' not in result


def test_no_verifier_baseline_runs_on_the_same_cohort():
    result=cross_validate(rows(),[],{})
    assert set(result['metrics'])=={'raw_generator','base_rate','calibrated_generator'}
    assert len(result['predictions']['calibrated_generator'])==30


def test_empty_gemini_reply_still_has_billable_reasoning():
    from vgx.common.billing import reconcile_usage
    usage=reconcile_usage({'prompt_tokens':262,'total_tokens':1366,
                           'completion_tokens_details':{'reasoning_tokens':1104}},model='google/gemini-3.5-flash-lite')
    assert usage.complete and usage.output_tokens==1104 and usage.reasoning_tokens==1104
    assert usage.basis=='total_minus_input_completion_omitted'
    invalid=reconcile_usage({'prompt_tokens':262,'total_tokens':263,
                             'completion_tokens_details':{'reasoning_tokens':1104}},model='google/gemini-3.5-flash-lite')
    assert not invalid.complete


def test_degenerate_conditional_correlations_are_missing_not_zero():
    from vgx.gpqa.signal_analysis import dependence_summary
    result=dependence_summary(rows(),['binary','probability'])
    assert result['0']['n']==10 and result['1']['n']==20
    assert all(pair['pearson'] is None for pair in result['0']['pairs'])


def test_tail_diagnostics_do_not_report_perfection_for_empty_tail():
    from vgx.gpqa.signal_analysis import tail_diagnostics
    result=tail_diagnostics([1,0],{'p':[.96,.5]})['p']
    assert result[0]['n']==1 and result[0]['accuracy']==1.
    assert result[1]['n']==0 and result[1]['accuracy'] is None


def test_likelihood_bootstrap_uses_two_bins_for_binary_and_three_for_probability():
    from vgx.gpqa.signal_analysis import likelihood_stability
    binary=likelihood_stability(rows(),'binary','binary',repeats=5)
    probability=likelihood_stability(rows(),'probability','probability',repeats=5)
    assert binary['counts_by_correctness']=={'0':[10,0],'1':[0,20]}
    assert probability['counts_by_correctness']=={'0':[10,0,0],'1':[0,0,20]}
    assert len(binary['likelihood_ratio_interval']['low'])==2
    assert len(probability['likelihood_ratio_interval']['low'])==3


def test_all_candidate_diagnostic_keeps_missing_signals_without_bayes_imputation():
    data=rows()
    for row in data[::4]:row['verifiers'].pop('binary')
    result=cross_validate(data,['binary'],{'binary':'binary'},allow_missing=True)
    assert result['n']==30 and result['missing_signal_handling']=='training_fold_median_and_indicator'
    assert 'joint:binary' in result['metrics'] and 'bayes:binary' not in result['metrics']
