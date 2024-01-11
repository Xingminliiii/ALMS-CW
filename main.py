from A.taskA import load_and_preprocess_data, build_model,\
    compile_and_train_model, evaluate_model, plot_results

def main():
    filepath = 'Datasets/pneumoniamnist.npz'
    x_train, y_train, x_val, y_val, x_test, y_test,\
        class_weights = load_and_preprocess_data(filepath)
    model = build_model()
    model, history = compile_and_train_model(model, x_train, y_train, x_val, y_val, class_weights)
    evaluate_model(model, x_test, y_test)
    plot_results(history)

if __name__ == '__main__':
    main()