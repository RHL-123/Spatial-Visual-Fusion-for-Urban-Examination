import json
import os
import sys

def main():
    base_dir = os.path.dirname(__file__)
    
    # 支持命令行传入输入文件路径，默认使用 500m 半径的文件
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = os.path.join(base_dir, '小区周边 POI 数据_高德_mapped_v2_500m.json')
    
    # 输出文件名根据输入文件名动态生成（在输入文件名后加_fixed）
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_fixed{ext}"

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    types_to_remove = ['汽车维修', '汽车服务', '汽车销售']
    
    modified_count = 0
    removed_pois_total = 0
    
    for community, info in data.items():
        community_modified = False
        
        # 1. 修剪 category_distribution
        cat_dist = info.get('category_distribution', {})
        for cat in types_to_remove:
            if cat in cat_dist:
                del cat_dist[cat]
                community_modified = True
                
        # 2. 彻底清洗 poi_list 中的脏数据
        original_pois = info.get('poi_list', [])
        clean_pois = []
        for poi in original_pois:
            poi_type = poi.get('type', '')
            is_noise = any(noise_cat in poi_type for noise_cat in types_to_remove)
            
            if not is_noise:
                clean_pois.append(poi)
            else:
                community_modified = True
                removed_pois_total += 1
                
        if community_modified:
            info['poi_list'] = clean_pois
            info['total_count'] = len(clean_pois)
            
            # 3. 重新计算并修剪 type_distribution
            new_type_dist = {}
            for poi in clean_pois:
                t = poi.get('type', '')
                if t:
                    new_type_dist[t] = new_type_dist.get(t, 0) + 1
            info['type_distribution'] = new_type_dist
            
            modified_count += 1

    print(f"清洗完成:")
    print(f"- 共有 {modified_count} 个社区发生变更。")
    print(f"- 从全量数据中彻底剥离了 {removed_pois_total} 个汽车相关的 POI 点位。")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

