from models.model_graph_mil import PatchGCN_Surv


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


if __name__ == "__main__":
    model_dict = {'num_layers': 4, 'edge_agg': 'spatial', 'resample': 0.0, 'n_classes': 4}
    model = PatchGCN_Surv(**model_dict)
    getModelSize(model)
