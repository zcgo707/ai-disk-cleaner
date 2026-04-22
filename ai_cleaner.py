import os
import pymilvus
from sentence_transformers import SentenceTransformer
import time

# --- 配置 Milvus 连接 ---
try:
    pymilvus.connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus.")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1) # 如果连接失败，退出程序

# --- 加载轻量级 Embedding 模型 ---
# 注意：首次加载模型会下载，可能需要一点时间
print("Loading embedding model...")
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    exit(1)

def scan_and_index(root_path, collection_name="file_index"):
    """
    扫描给定路径下的文件，提取元数据并存入 Milvus。
    """
    # 1. 创建集合 (如果不存在)
    if collection_name not in pymilvus.list_collections():
        print(f"Collection '{collection_name}' does not exist. Creating it...")
        # 定义字段
        path_field = pymilvus.FieldSchema("path", dtype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=65535) # 增大 max_length
        size_field = pymilvus.FieldSchema("size", dtype=pymilvus.DataType.INT64)
        vector_field = pymilvus.FieldSchema("vector", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=384) # MiniLM 维度
        
        # 创建 Schema
        schema = pymilvus.CollectionSchema(
            fields=[path_field, size_field, vector_field],
            description="File Metadata"
        )
        
        # 创建 Collection
        collection = pymilvus.Collection(collection_name, schema)
        
        # 创建索引以加速搜索
        # HNSW 是一种常用的高效索引类型
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2", # L2 距离
            "params": {"M": 8, "efConstruction": 200}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"Collection '{collection_name}' created and index built.")
    else:
        print(f"Collection '{collection_name}' already exists. Loading it...")
        collection = pymilvus.Collection(collection_name)

    # 2. 遍历文件系统
    print(f"正在扫描: {root_path}")
    data_to_insert = {'path': [], 'size': [], 'vector': []} # 使用字典格式存储批量数据
    count = 0
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 获取属性
                stat_info = os.stat(file_path)
                size = stat_info.st_size
                
                # 过滤掉小于 1KB 的文件，减少数据量和噪声
                if size < 1024: 
                    continue 
                
                # 计算嵌入向量 (这里对路径进行编码，也可以编码内容，但编码路径更快)
                # 注意：路径字符串可能很长，编码效果取决于模型
                # 如果需要基于内容的相似性，需要读取文件内容并编码
                embedding = embed_model.encode(file_path).tolist() 
                
                # 添加到待插入列表
                data_to_insert['path'].append(file_path)
                data_to_insert['size'].append(size)
                data_to_insert['vector'].append(embedding)
                
                count += 1
                
                # 批量插入，提高效率
                if len(data_to_insert['path']) >= 1000:
                    print(f"Inserting batch of {len(data_to_insert['path'])} items... ({count} total processed so far)")
                    collection.insert([data_to_insert['path'], data_to_insert['size'], data_to_insert['vector']])
                    
                    # 重置列表
                    data_to_insert = {'path': [], 'size': [], 'vector': []}
                    
            except (OSError, PermissionError) as e:
                # 跳过无法访问的文件或符号链接等问题
                print(f"Skipping file due to error: {file_path}, Error: {e}")
                continue # 继续处理下一个文件
            except Exception as e:
                print(f"Unexpected error processing file: {file_path}, Error: {e}")
                # 可以选择跳过或中断，这里选择跳过
                continue 

    # 插入剩余数据
    if data_to_insert['path']:
        print(f"Inserting final batch of {len(data_to_insert['path'])} items...")
        collection.insert([data_to_insert['path'], data_to_insert['size'], data_to_insert['vector']])

    # 刷新集合，确保数据写入
    collection.flush()
    
    # 加载到内存以供搜索
    print("Loading collection into memory for searching...")
    collection.load()
    print(f"Scan & Index completed! Processed {count} files into collection '{collection_name}'.")

# --- 运行扫描 (例如扫描 /mnt/d 盘，根据你的实际路径修改) ---
# 注意：Milvus 服务必须在 localhost:19530 上运行
# 请将 "/mnt/d" 替换为你想要扫描的实际路径
ROOT_PATH_TO_SCAN = "/mnt/c" # 修改为你想扫描的路径，例如 "/home/user/Documents" 或 "C:\\" (在 WSL 中可能是 "/mnt/c")

if __name__ == "__main__":
    scan_and_index(ROOT_PATH_TO_SCAN)
