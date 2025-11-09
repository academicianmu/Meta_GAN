from graphviz import Digraph

# 初始化有向图（设置布局、节点/边样式）
dot = Digraph(
    name="meta_gan_unet_style",
    format="png",
    graph_attr={"rankdir": "TB", "bgcolor": "transparent"},  # TB：从上到下布局
    node_attr={"shape": "box3d", "style": "filled"},         # 3D块样式
    edge_attr={"penwidth": "2", "color": "gray"}             # 边样式
)

# ========== 1. 元学习任务流（顶部） ==========
dot.node("task_source", "元任务数据源", fillcolor="lightblue")
dot.node("task_sample", "子任务采样", fillcolor="lightblue")
dot.node("meta_loop", "多子任务循环", fillcolor="lightblue")
dot.edge("task_source", "task_sample")
dot.edge("task_sample", "meta_loop")

# ========== 2. 生成器（编码+解码，模拟U-Net） ==========
# 编码路径（左→右，维度递减）
dot.node("gen_noise", "噪声输入\nz~N(0,1)", fillcolor="lightgreen")
dot.node("gen_enc1", "FC+ReLU\ndim=256", fillcolor="lightgreen")
dot.node("gen_enc2", "FC+ReLU\ndim=512", fillcolor="lightgreen")
dot.node("gen_bottleneck", "瓶颈层", fillcolor="lightgreen")
dot.edge("gen_noise", "gen_enc1")
dot.edge("gen_enc1", "gen_enc2")
dot.edge("gen_enc2", "gen_bottleneck")

# 解码路径（右→左，维度递增 + 跳跃连接）
dot.node("gen_dec1", "FC+ReLU\ndim=512", fillcolor="lightgreen")
dot.node("gen_dec2", "FC+ReLU\ndim=256", fillcolor="lightgreen")
dot.node("gen_out", "生成数据\nx_hat", fillcolor="lightgreen")
dot.edge("gen_bottleneck", "gen_dec1")
dot.edge("gen_dec1", "gen_dec2")
dot.edge("gen_dec2", "gen_out")
# 跳跃连接（编码→解码）
dot.edge("gen_enc2", "gen_dec1", color="red", penwidth="3")  # 红色强调跳跃连接
dot.edge("gen_enc1", "gen_dec2", color="red", penwidth="3")

# ========== 3. 判别器（编码+解码，模拟U-Net） ==========
# 编码路径
dot.node("dis_input", "数据输入\nx/x_hat", fillcolor="lightcoral")
dot.node("dis_enc1", "FC+LeakyReLU\ndim=512", fillcolor="lightcoral")
dot.node("dis_enc2", "FC+LeakyReLU\ndim=256", fillcolor="lightcoral")
dot.node("dis_bottleneck", "瓶颈层", fillcolor="lightcoral")
dot.edge("dis_input", "dis_enc1")
dot.edge("dis_enc1", "dis_enc2")
dot.edge("dis_enc2", "dis_bottleneck")

# 解码路径 + 跳跃连接
dot.node("dis_dec1", "FC+LeakyReLU\ndim=256", fillcolor="lightcoral")
dot.node("dis_dec2", "FC+LeakyReLU\ndim=512", fillcolor="lightcoral")
dot.node("dis_out", "真假概率", fillcolor="lightcoral")
dot.edge("dis_bottleneck", "dis_dec1")
dot.edge("dis_dec1", "dis_dec2")
dot.edge("dis_dec2", "dis_out")
dot.edge("dis_enc2", "dis_dec1", color="red", penwidth="3")
dot.edge("dis_enc1", "dis_dec2", color="red", penwidth="3")

# ========== 4. 元更新（连接任务与模型） ==========
dot.node("gen_meta_upd", "生成器元更新", fillcolor="orange")
dot.node("dis_meta_upd", "判别器元更新", fillcolor="orange")
dot.edge("gen_out", "gen_meta_upd")
dot.edge("dis_out", "dis_meta_upd")
dot.edge("gen_meta_upd", "meta_loop")
dot.edge("dis_meta_upd", "meta_loop")

# 保存并渲染图像
dot.render("meta_gan_unet_style", view=True)  # 保存为PNG并打开查看