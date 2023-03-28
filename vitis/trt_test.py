def create_torch_model(MODEL_WEIGHTS='/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth',v=False):
    import json
    import trt_pose.coco
    import trt_pose.models
    from torchinfo import summary
    import torch


    with open('/trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).eval()

    #MODEL_WEIGHTS = '/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'

    model.load_state_dict(torch.load(MODEL_WEIGHTS,map_location=torch.device('cpu')))

    if v:
        print(summary(model))

    return model
