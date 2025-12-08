# Configs

Here you can find a collection of configurations for anemoi models used in scientific experiments towards increasing time resolution. This particular branch of anemoi-configs was created to facilitate sharing configs, provide an overview and keep track of experiments which are performed at various institutions involved. It is not planned for the moment to converge with the main branch. 

For the moment models of three different types are considered:
 - vanilla models: forecaster with 1 output step per model forward, i.e. horizon = step
 - multi-out models: forecaster with n output steps per model forward, i.e. horizon = n*step
 - time-interpolator: creates n intermediate time steps when provided with two time steps of a forecaster, i.e step = horizon/n

For the multi-out models it is strongly suggested to perform scientific experiments with the anemoi-core branch https://github.com/ecmwf/anemoi-core/tree/multi-out-tmp , which is a functioning branch that has been frozen while https://github.com/ecmwf/anemoi-core/tree/feat/multi-output-steps is a branch that is undergoing continuing development and hence less suited for scientific experiments and comparisons. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.