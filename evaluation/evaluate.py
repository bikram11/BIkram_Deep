import argparse
import json
import numpy as np

def validate(sub, ans, meta):
    intersection = set(ans).intersection(set(sub)).intersection(set(meta['weights']))
    print('Evaluate {} scenes.\n'.format(len(intersection)))
    if not len(intersection):
        return 1
    for k in intersection:
        if len(sub[k])!=len(ans[k]):
            print('There is length mismatch in scene {} (sub and ans)'.format(k))
            return 1

    return 0

def spd_abs_error(sub, ans, meta):
    all_error = 0
    weights = 0
    test_scenes = set(ans).intersection(set(sub)).intersection(set(meta['weights']))
    print('Errors: ')
    for test_scene in sorted(test_scenes):
        pr = np.array(sub[test_scene][meta['first_frame']-1:])
        gt = np.array(ans[test_scene][meta['first_frame']-1:])
        limit = meta['limit_gradient']*gt + meta['limit_intercept']
        err = np.minimum(np.abs((pr - gt)/limit), 1).mean()
        print('  {}: {} (weight: {})'.format(test_scene, err, meta['weights'][test_scene]))
        all_error += meta['weights'][test_scene]*err
        weights += meta['weights'][test_scene]

    return all_error/weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth-path', default = './data/ans.json')
    parser.add_argument('--predictions-path', default = './data/sub.json')
    parser.add_argument('--meta-data-path', default = './data/meta.json')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(args.ground_truth_path) as f:
        ans = json.load(f)
    with open(args.predictions_path) as f:
        sub = json.load(f)
    with open(args.meta_data_path) as f:
        meta = json.load(f)

    status = validate(sub, ans, meta)
    if status == 0:
        error = spd_abs_error(sub, ans, meta)
        print('\nTotal Error: {}'.format(error))

if __name__ == '__main__':
    main()
