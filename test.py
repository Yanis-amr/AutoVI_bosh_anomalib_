from utils import *




def parse_arguments():
    """
        Parse user arguments for the evaluation of a method on the AutoVI
        dataset.

        returns:
            Parsed user arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment_name',
        default=None,
        required=True,
        help='Custom name for the experiment')

    parser.add_argument(
        '--class_name',
        default=None,
        required=True,
        help='Name of the class (class folder name under ./datasets)')

    parser.add_argument(
        '--model_name',
        default='Padim',
        required=False,
        help='Anamoly detection model'
             'use Padim model if not specified')

    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()
    output_dir = "./results/" + args.experiment_name + "/" + args.model_name + "/" + args.class_name
    num_tests = 8
    try:
        metrics_all = [pd.read_excel(output_dir + "/metrics.xlsx", index_col=0)]
    except:
        metrics_all = []
    try:
        results_all = [pd.read_json(output_dir + "/cls_results.json")]
    except:
        results_all = []
    for i in range(num_tests):

        ## init datamodule from folder
        datamodule = init_datamodule(args.class_name)

        ## reset the model at each test
        if args.model_name == 'Padim':
            model = Padim() 
        elif args.model_name == 'Patchcore':
            model = Patchcore() ## reset the model at each test
        elif args.model_name == 'EfficientAd':
            model = EfficientAd() ## reset the model at each test
        elif args.model_name == 'Draem':
            model = Draem() ## reset the model at each test
        elif args.model_name == 'Dsr':
            model = Dsr() ## reset the model at each test
        elif args.model_name == 'Cflow':
            model = Cflow() ## reset the model at each test
        else:
            break

        ## start training
        t0 = time.time()
        engine = train(datamodule, model, "./results/" + args.experiment_name)
        train_time = time.time() - t0
        ## predict cls and segm results on test dataset
        predictions = predict(datamodule, engine)

        ## compute and save metrics
        metrics = compute_metrics(predictions, output_dir + "/latest/")
        metrics['train_time'] = [train_time]
        metrics_all.append(metrics)
        metrics = pd.concat(metrics_all, ignore_index = True)
        metrics.to_excel(output_dir + "/metrics.xlsx")


        result = cls_result(predictions)
        results_all.append(result)
        results = pd.concat(results_all, ignore_index = True)
        results.to_json(output_dir + "/cls_results.json")
        torch.cuda.empty_cache()

        
if __name__ == "__main__":
    main()
