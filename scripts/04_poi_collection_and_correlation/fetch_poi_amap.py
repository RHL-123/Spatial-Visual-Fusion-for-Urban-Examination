"""
从高德地图API爬取老旧小区周边POI数据（增强版）
=============================================
核心逻辑：
  - 坐标来源: ground_truth_final_neighborhood_filtered.xlsx 的「谷歌经纬度」按社区取均值
  - API: 高德地图 Around Search (周边搜索)
  - 搜索半径: 1500m
  - 查询方式: 7大类别分别查询，每类翻页（最多10页×25条=250条/类）
  - 去重: 按POI的id跨类别去重
  - 类别映射: 高德三级类型 -> 7大目标类别（医疗保健服务→医疗保健，科教文化服务→科教文化）
  - 噪音过滤: 过滤不在目标类别的类型（摩托车服务、商务住宅等）
  - 输出: 小区周边POI数据_高德_mapped_v2.json（含center_location字段记录爬取中心点）
"""

import json
import time
import requests
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ===== 配置 =====
AMAP_KEY = input('请输入高德API Key: ').strip()
SEARCH_RADIUS = 500
PAGE_SIZE = 25
# 输出文件名包含搜索半径信息，便于区分不同半径的爬取结果
OUTPUT_FILE = f'小区周边 POI 数据_高德_mapped_v2_{SEARCH_RADIUS}m.json'

GROUND_TRUTH_EXCEL = Path(__file__).parent.parent / 'ground_truth_final_neighborhood_filtered.xlsx'
AROUND_SEARCH_URL = 'https://restapi.amap.com/v3/place/around'

# 7大目标类别（高德 types 参数用中文类别名）
TARGET_CATEGORIES = ['餐饮服务', '生活服务', '购物服务', '公司企业', '医疗保健服务', '科教文化服务', '住宿服务']

# 高德大类名 -> 输出类别名的映射（"医疗保健服务"输出为"医疗保健"，"科教文化服务"输出为"科教文化"）
CATEGORY_RENAME = {
    '医疗保健服务': '医疗保健',
    '科教文化服务': '科教文化',
}

# API Key 错误标识（高德常见错误码）
KEY_ERROR_INFOS = {
    'INVALID_USER_KEY', 'USER_KEY_RECYCLED',
    'SERVICE_NOT_AVAILABLE', 'INVALID_USER_SCODE',
    'USER_DAILY_QUERY_OVER_LIMIT', 'USER_ABROAD_DAILY_QUERY_OVER_LIMIT',
}


class AmapKeyError(Exception):
    """高德 API Key 失效或额度耗尽，应停止爬取。"""
    pass


# 不在7大目标类别中的高德大类（原始数据中出现过但被过滤）
FILTERED_OUT_TYPES = {'体育休闲服务', '摩托车服务', '金融保险服务', '风景名胜',
                      '公共设施', '政府机构及社会团体', '交通设施服务', '道路附属设施',
                      '地名地址信息', '商务住宅'}


def load_community_centers(excel_path):
    """从 Excel 按社区分组，取楼栋坐标均值作为社区中心。"""
    df = pd.read_excel(excel_path)
    def parse_coord(s):
        parts = str(s).split(',')
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return None, None
    df['lng'], df['lat'] = zip(*df['谷歌经纬度'].apply(parse_coord))
    df = df.dropna(subset=['lng', 'lat'])
    centers = df.groupby('社区').agg(lng=('lng', 'mean'), lat=('lat', 'mean')).reset_index()
    return {row['社区']: f"{row['lng']:.8f},{row['lat']:.8f}" for _, row in centers.iterrows()}


def fetch_category(location, category_name, radius=SEARCH_RADIUS, max_pages=10):
    """
    按单个类别查询周边POI，支持翻页。
    遇到 Key 错误时抛出 AmapKeyError 让上层中断整个爬取流程。
    """
    all_cat_pois = []
    page = 1
    
    while page <= max_pages:
        params = {
            'key': AMAP_KEY,
            'location': location,
            'radius': radius,
            'offset': PAGE_SIZE,
            'page': page,
            'types': category_name,
            'extensions': 'all',
            'output': 'JSON',
        }
        try:
            resp = requests.get(AROUND_SEARCH_URL, params=params, timeout=15)
            data = resp.json()
            if data.get('status') != '1':
                info = data.get('info', 'UNKNOWN_ERROR')
                # Key 失效 / 额度耗尽 → 抛异常，不要保存当前社区
                if info in KEY_ERROR_INFOS:
                    raise AmapKeyError(f'API Key 错误: {info} ({data.get("infocode", "")})')
                # 其他错误（如无结果）→ 正常退出
                print(f'    [WARN] {category_name} 第{page}页: {info}')
                break
            
            pois = data.get('pois', [])
            all_cat_pois.extend(pois)
            
            # 如果当前页返回的POI数量少于 PAGE_SIZE，说明已经是最后一页了，终止翻页
            if len(pois) < PAGE_SIZE:
                break
            
            page += 1
            time.sleep(0.2)  # 给API稍微留一点请求间隔
        except AmapKeyError:
            raise  # Key 错误必须向上传播，不能吞掉
        except Exception as e:
            print(f'    [WARN] {category_name} 第{page}页: {e}')
            break
            
    return all_cat_pois


def aggregate_poi_data(poi_dict):
    """将POI列表聚合为原始JSON格式的统计结果，按POI的id去重。"""
    seen_ids = set()
    deduped_pois = []
    type_dist = defaultdict(int)
    cat_dist = defaultdict(int)

    # 按原始类别的出现顺序，构建最终平级的category_distribution
    for cat_name, pois in poi_dict.items():
        for poi in pois:
            poi_id = poi.get('id', '')
            if poi_id and poi_id in seen_ids:
                continue
            if poi_id:
                seen_ids.add(poi_id)
            deduped_pois.append(poi)
            
            # type_distribution: 原始三级type原样计数
            amap_type = poi.get('type', '')
            if amap_type:
                type_dist[amap_type] += 1
            
            # category_distribution: 取POI实际的一级大类，过滤噪音类型
            top_cat = amap_type.split('|')[0].split(';')[0] if amap_type else ''
            # 跳过噪音类型
            if top_cat in FILTERED_OUT_TYPES:
                continue
            output_cat = CATEGORY_RENAME.get(top_cat, top_cat)
            cat_dist[output_cat] += 1

    return {
        'total_count': len(deduped_pois),
        'type_distribution': dict(type_dist),
        'category_distribution': dict(cat_dist),
        'poi_list': deduped_pois,
    }


def main():
    MAX_PAGES = 10  # 每类最多翻10页
    print(f'读取楼栋坐标: {GROUND_TRUTH_EXCEL}')
    community_centers = load_community_centers(GROUND_TRUTH_EXCEL)
    print(f'共 {len(community_centers)} 个社区')
    print(f'查询方式: 7大类别分别查询，每类翻页至多{MAX_PAGES}页(每页{PAGE_SIZE}条)')
    print()

    output_path = Path(__file__).parent / OUTPUT_FILE
    result = {}

    # 如果已有部分结果，先加载（支持断点续爬）
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        print(f'发现已有输出文件，已加载 {len(result)} 个社区的数据')

    for i, (name, location) in enumerate(community_centers.items()):
        # 跳过已爬取且有数据的社区
        # 注意：不能用"7类齐全"做判断，因为有些社区周边确实不全
        if name in result and result[name].get('total_count', 0) > 0:
            print(f'[{i+1}/{len(community_centers)}] {name}  已存在，跳过')
            continue

        print(f'[{i+1}/{len(community_centers)}] {name}')

        # 7大类别分别查询
        try:
            poi_dict = {}
            for cat in TARGET_CATEGORIES:
                pois = fetch_category(location, cat)
                output_name = CATEGORY_RENAME.get(cat, cat)
                poi_dict[output_name] = pois
                time.sleep(0.2)
        except AmapKeyError as e:
            print(f'\n[FATAL] {e}')
            print('Key 失效，当前社区数据未保存，请更换 Key 后重新运行（会续爬此社区）')
            return

        aggregated = aggregate_poi_data(poi_dict)
        # 将本次爬取所用的中心点坐标也写进结果
        aggregated['center_location'] = location
        result[name] = aggregated
        print(f'  中心={location}  POI={aggregated["total_count"]}条')

        # 每爬完一个社区立即保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\n完成！保存至: {output_path}')


if __name__ == '__main__':
    main()

