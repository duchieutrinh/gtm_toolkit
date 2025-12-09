import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import time
import re
import os
import unidecode
from rapidfuzz import fuzz
import streamlit as st

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
import networkx as nx
import io

# =============================================================================
# TÌM KIẾM $$$
# =============================================================================
def show_adsense_ad(publisher_id, slot_id, ad_format="auto", is_responsive=True):
    """
    Tạo và hiển thị một khối quảng cáo Google AdSense.

    Args:
        publisher_id (str): ID nhà xuất bản của bạn (vd: "ca-pub-1234567890123456").
        slot_id (str): ID của đơn vị quảng cáo (vd: "1234567890").
        ad_format (str): Định dạng quảng cáo, mặc định là "auto".
        is_responsive (bool): Quảng cáo có đáp ứng hay không, mặc định là True.
    """
    ad_script = f"""
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={publisher_id}"
             crossorigin="anonymous"></script>
        <!-- Tên đơn vị quảng cáo của bạn -->
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="{publisher_id}"
             data-ad-slot="{slot_id}"
             data-ad-format="{ad_format}"
             data-full-width-responsive="{str(is_responsive).lower()}"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({{}});
        </script>
    """
    st.components.v1.html(ad_script, height=100) # Có thể điều chỉnh chiều cao nếu cần

# --- Hằng số ---
EARTH_RADIUS_METERS = 6371000.0

# =============================================================================
# CÁC HÀM LOGIC (TÁI SỬ DỤNG TỪ PHIÊN BẢN DESKTOP)
# Chúng ta có thể sao chép gần như nguyên vẹn các hàm này.
# =============================================================================

def normalize_name(text, abbreviation_map):
    if pd.isna(text): return ""
    normalized = unidecode.unidecode(str(text)).lower()
    if abbreviation_map:
        pattern = r'\b(' + '|'.join(re.escape(key) for key in abbreviation_map.keys()) + r')\b'
        normalized = re.sub(pattern, lambda m: abbreviation_map.get(m.group(1), m.group(1)), normalized)
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def validate_lat_lon(df, lat_col='latitude', lon_col='longitude'):
    condition_out_of_range = (
        (~df[lat_col].between(-90, 90, inclusive='both')) |
        (~df[lon_col].between(-180, 180, inclusive='both'))
    )
    rows_to_invalidate = df[df['is_valid'] & condition_out_of_range]
    if not rows_to_invalidate.empty:
        df.loc[rows_to_invalidate.index, 'is_valid'] = False
    return df, len(rows_to_invalidate)

def clean_lat_lon(df, lat_col='latitude', lon_col='longitude'):
    df[lat_col] = pd.to_numeric(df[lat_col].astype(str).str.strip(), errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col].astype(str).str.strip(), errors='coerce')
    nan_rows = df[df[lat_col].isna() | df[lon_col].isna()]
    if not nan_rows.empty:
        df.loc[nan_rows.index, 'is_valid'] = False
    return df, len(nan_rows)

def run_processing_logic(config, status_callback, progress_callback):
    try:
        start_time = time.time()
        lat_col = config['lat_col']
        lon_col = config['lon_col']

        # 1. Đọc File từ đối tượng bytes trong Streamlit
        read_dtype = str if config.get('read_as_text', False) else None
        if read_dtype: status_callback("Chế độ đọc dưới dạng văn bản được bật.")
        
        status_callback(f"Đang đọc file gốc: {config['file1_name']}...")
        df1 = pd.read_excel(config['file1_data'], dtype=read_dtype)
        
        status_callback(f"Đang đọc file so sánh: {config['file2_name']}...")
        df2 = pd.read_excel(config['file2_data'], dtype=read_dtype)
        
        status_callback("Đọc file hoàn tất.")

        # Kiểm tra cột
        if lat_col not in df1.columns or lon_col not in df1.columns:
            raise ValueError(f"Cột vĩ độ '{lat_col}' hoặc kinh độ '{lon_col}' không tồn tại trong file gốc.")
        if config.get('enable_name_matching'):
            name_col1 = config['name_col1']
            name_col2 = config['name_col2']
            if name_col1 not in df1.columns: raise ValueError(f"Cột tên '{name_col1}' không tồn tại trong file gốc.")
            if name_col2 not in df2.columns: raise ValueError(f"Cột tên '{name_col2}' không tồn tại trong file so sánh.")

        # Khởi tạo các cột mới trong df2
        COLUMN_MAPPING = config['column_mapping']
        new_cols_df2 = list(COLUMN_MAPPING.keys()) + ['Khoảng cách']
        for col in new_cols_df2:
            if col not in df2.columns:
                df2[col] = pd.NA

        # 2. Chuẩn bị dữ liệu
        status_callback("Đang chuẩn bị và xác thực dữ liệu...")
        for df, name in [(df1, 'File 1'), (df2, 'File 2')]:
            for col in [lat_col, lon_col]:
                if col not in df.columns: df[col] = np.nan
            df['is_valid'] = True
            df, nan_count = clean_lat_lon(df, lat_col=lat_col, lon_col=lon_col)
            if nan_count > 0: status_callback(f"[{name}] Đã đánh dấu {nan_count} dòng có lat/long thiếu hoặc không hợp lệ.")
            df, invalid_range_count = validate_lat_lon(df, lat_col=lat_col, lon_col=lon_col)
            if invalid_range_count > 0: status_callback(f"[{name}] Đã đánh dấu {invalid_range_count} dòng có lat/long ngoài khoảng cho phép.")

        df1_valid = df1[df1['is_valid']].copy()
        df2_valid = df2[df2['is_valid']].copy()
        status_callback(f"File 1 có {len(df1_valid)}/{len(df1)} dòng hợp lệ.")
        status_callback(f"File 2 có {len(df2_valid)}/{len(df2)} dòng hợp lệ.")

        # Chuẩn hóa tên
        if config.get('enable_name_matching'):
            status_callback("Tối ưu: Chuẩn hóa trước các cột tên...")
            progress_callback(5, "Chuẩn hóa tên...")
            abbreviation_map = config.get('abbreviation_map', {})
            df1_valid['normalized_name'] = df1_valid[config['name_col1']].apply(lambda x: normalize_name(x, abbreviation_map))
            df2_valid['normalized_name'] = df2_valid[config['name_col2']].apply(lambda x: normalize_name(x, abbreviation_map))

        # 3. Xử lý BallTree
        if df1_valid.empty or df2_valid.empty:
            raise ValueError("Một trong hai file không có dữ liệu vị trí hợp lệ để xử lý.")

        status_callback("Chuyển đổi tọa độ sang radians...")
        coords1 = np.radians(df1_valid[[lat_col, lon_col]].values)
        coords2 = np.radians(df2_valid[[lat_col, lon_col]].values)

        status_callback("Tạo cây dữ liệu BallTree từ file gốc...")
        tree = BallTree(coords1, metric='haversine')
        
        radius = (config['distance_m'] + 1) / EARTH_RADIUS_METERS
        status_callback(f"Truy vấn các điểm trong bán kính {config['distance_m']}m...")
        progress_callback(10, "Xây dựng cây và truy vấn...")
        indices, distances = tree.query_radius(coords2, r=radius, return_distance=True)
        status_callback("Truy vấn hoàn tất.")

        # 4. Xử lý và cập nhật kết quả (Vector hóa)
        status_callback("Đang xử lý kết quả (thuật toán vector hóa)...")
        progress_callback(20, "Xử lý kết quả...")
        
        lens = [len(i) for i in indices]
        df2_idx_list = np.repeat(df2_valid.index.to_numpy(), lens)

        if len(df2_idx_list) == 0:
            status_callback("Không tìm thấy cặp điểm nào trong bán kính đã cho.")
        else:
            df1_iloc_list = np.concatenate(indices)
            dist_list = np.concatenate(distances)

            pairs_df = pd.DataFrame({
                'df2_index': df2_idx_list,
                'df1_valid_iloc': df1_iloc_list,
                'distance_rad': dist_list
            })
            pairs_df['df1_index'] = df1_valid.index[pairs_df['df1_valid_iloc']]
            progress_callback(40, "Đã tạo các cặp tiềm năng...")

            if config.get('enable_name_matching'):
                status_callback("Thực hiện so khớp tên hàng loạt...")
                pairs_df['name1'] = pairs_df['df1_index'].map(df1_valid['normalized_name'])
                pairs_df['name2'] = pairs_df['df2_index'].map(df2_valid['normalized_name'])
                scores = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(pairs_df['name1'], pairs_df['name2'])]
                pairs_df['score'] = scores
                
                threshold = config['name_match_threshold']
                pairs_df = pairs_df[pairs_df['score'] >= threshold].copy()
                status_callback(f"Đã lọc, còn lại {len(pairs_df)} cặp sau khi so khớp tên.")
            
            progress_callback(70, "Đã so khớp tên (nếu có)...")

            if not pairs_df.empty:
                status_callback("Chọn cặp điểm tốt nhất cho mỗi điểm...")
                pairs_df.sort_values(by=['df2_index', 'distance_rad'], inplace=True)
                best_matches_df = pairs_df.drop_duplicates(subset='df2_index', keep='first').copy()
                
                status_callback(f"Tìm thấy {len(best_matches_df)} điểm trùng duy nhất. Chuẩn bị cập nhật...")
                progress_callback(85, "Đang chuẩn bị cập nhật...")

                update_data = df1.loc[best_matches_df['df1_index'], list(COLUMN_MAPPING.values())].copy()
                update_data.rename(columns={v: k for k, v in COLUMN_MAPPING.items()}, inplace=True)
                update_data['Khoảng cách'] = (best_matches_df['distance_rad'].values * EARTH_RADIUS_METERS).round(2).astype(str) + 'm'
                update_data.index = best_matches_df['df2_index'].values
                
                status_callback("Cập nhật dữ liệu vào file so sánh...")
                df2.update(update_data)
            else:
                status_callback("Không có cặp điểm nào thỏa mãn điều kiện sau khi lọc.")

        # 5. Xuất kết quả
        status_callback("Đang tạo file kết quả để tải về...")
        progress_callback(95, "Đang tạo file kết quả...")
        
        # Chuyển đổi DataFrame thành file Excel trong bộ nhớ
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            df2.to_excel(writer, index=False, sheet_name='Sheet1')
        output_buffer.seek(0)

        end_time = time.time()
        status_callback(f"Hoàn thành! Tổng thời gian: {end_time - start_time:.2f} giây.")
        progress_callback(100, "Hoàn thành!")
        
        return output_buffer

    except Exception as e:
        status_callback(f"LỖI: {e}", is_error=True)
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
        return None

def create_analysis_map(df_all, df_results, config):
    """Tạo đối tượng bản đồ Folium (không lưu file)."""
    if not FOLIUM_AVAILABLE:
        raise ImportError("Vui lòng cài đặt thư viện 'folium' để sử dụng tính năng bản đồ.")

    unique_routes = df_all[config['route_col']].dropna().unique()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    route_colors = {route: colors[i % len(colors)] for i, route in enumerate(unique_routes)}

    lat_center = df_all[config['lat_col']].mean()
    lon_center = df_all[config['lon_col']].mean()
    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles="CartoDB positron")

    for _, row in df_all.iterrows():
        route = row[config['route_col']]
        color = route_colors.get(route, 'gray')
        extra_info_html = ""
        for col in config.get('extra_cols', []):
            if col in row and pd.notna(row[col]):
                extra_info_html += f"<b>{col}:</b> {row[col]}<br>"
        tooltip_cols = config.get('tooltip_cols', [config['name_col']])
        tooltip_parts = [str(row[col]) for col in tooltip_cols if col in row and pd.notna(row[col])]
        tooltip_text = " - ".join(tooltip_parts) + f" ({route})"
        popup_html = f"<b>{row[config['name_col']]}</b><br><b>ID:</b> {row[config['id_col']]}<br><b>Tuyến:</b> {route}<br>{extra_info_html}"
        folium.CircleMarker(
            location=[row[config['lat_col']], row[config['lon_col']]],
            radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=350), tooltip=tooltip_text
        ).add_to(m)

    for _, row in df_results.iterrows():
        loc1 = [row['lat_goc'], row['lon_goc']]
        loc2 = [row['lat_hang_xom'], row['lon_hang_xom']]
        extra_info_line_html = ""
        for col in config.get('extra_cols', []):
            if col in row and pd.notna(row[col]):
                extra_info_line_html += f"<b>{col}:</b> {row[col]}<br>"
        popup_line = f"<u><b>Điểm gốc:</b> {row[f'{config['name_col']}_Gốc']} ({row[f'{config['route_col']}_Gốc']})</u><br>{extra_info_line_html}<hr style='margin: 5px 0;'><b>Hàng xóm khác tuyến:</b> {row['Hàng_xóm_gần_nhất_Tên']} ({row['Hàng_xóm_gần_nhất_Tuyến']})<br><b>Khoảng cách:</b> {row['Khoảng_cách(m)']}m"
        tooltip_line = f"Từ: {row[f'{config['name_col']}_Gốc']} -> {row['Hàng_xóm_gần_nhất_Tên']}"
        folium.PolyLine([loc1, loc2], color="red", weight=1.5, opacity=0.8, popup=folium.Popup(popup_line, max_width=350), tooltip=tooltip_line).add_to(m)

    return m

def run_route_analysis_logic(config, status_callback, progress_callback):
    """
    Logic cho Phân tích Tuyến đường (phiên bản Streamlit).
    """
    try:
        start_time = time.time()
        status_callback("Bắt đầu quá trình phân tích tuyến đường...")
        
        status_callback(f"Đang đọc file: {config['file_name']}...")
        df = pd.read_excel(config['file_data'], dtype=str)

        if config.get('filter_routes'):
            status_callback(f"Lọc dữ liệu theo các tuyến: {', '.join(config['filter_routes'])}")
            df = df[df[config['route_col']].isin(config['filter_routes'])].copy()
            if df.empty: raise ValueError("Không có dữ liệu nào khớp với các tuyến đã lọc.")
            status_callback(f"Còn lại {len(df)} điểm sau khi lọc.")

        required_cols = [config['lat_col'], config['lon_col'], config['route_col'], config['id_col'], config['name_col']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: raise ValueError(f"Các cột sau không tồn tại trong file: {', '.join(missing_cols)}")

        status_callback("Đang chuẩn bị và xác thực dữ liệu...")
        df['is_valid'] = True
        df, nan_count = clean_lat_lon(df, lat_col=config['lat_col'], lon_col=config['lon_col'])
        if nan_count > 0: status_callback(f"Đã đánh dấu {nan_count} dòng có lat/long không hợp lệ.")
        df, invalid_range_count = validate_lat_lon(df, lat_col=config['lat_col'], lon_col=config['lon_col'])
        if invalid_range_count > 0: status_callback(f"Đã đánh dấu {invalid_range_count} dòng có lat/long ngoài khoảng.")

        df_valid = df[df['is_valid']].copy()
        df_valid.reset_index(inplace=True)
        status_callback(f"Tìm thấy {len(df_valid)}/{len(df)} điểm có vị trí hợp lệ để phân tích.")
        if df_valid.empty: raise ValueError("Không có dữ liệu hợp lệ để phân tích.")

        status_callback("Xây dựng cây dữ liệu không gian...")
        coords = np.radians(df_valid[[config['lat_col'], config['lon_col']]].values.astype(float))
        tree = BallTree(coords, metric='haversine')

        radius_rad = config['distance_m'] / EARTH_RADIUS_METERS
        status_callback(f"Truy vấn các điểm trong bán kính {config['distance_m']}m...")
        indices, distances = tree.query_radius(coords, r=radius_rad, return_distance=True)

        status_callback("Phân tích sự bất hợp lý trong phân tuyến...")
        results_list = []
        total_points = len(df_valid)
        progress_callback(10, f"Đang phân tích {total_points} điểm...")

        for i in range(total_points):
            if i % 200 == 0: progress_callback(10 + int((i/total_points)*70), f"Đã xử lý {i}/{total_points} điểm...")
            p_original_index = df_valid.loc[i, 'index']
            p_route = df.loc[p_original_index, config['route_col']]
            if pd.isna(p_route): continue

            inconsistent_neighbors = []
            for j_idx, q_iloc in enumerate(indices[i]):
                if i == q_iloc: continue
                q_original_index = df_valid.loc[q_iloc, 'index']
                q_route = df.loc[q_original_index, config['route_col']]
                if pd.isna(q_route) or p_route == q_route: continue
                
                dist_m = distances[i][j_idx] * EARTH_RADIUS_METERS
                inconsistent_neighbors.append({'q_original_index': q_original_index, 'distance': dist_m})

            if inconsistent_neighbors:
                closest = min(inconsistent_neighbors, key=lambda x: x['distance'])
                others = sorted([n for n in inconsistent_neighbors if n != closest], key=lambda x: x['distance'])
                others_str = ", ".join([f"{df.loc[n['q_original_index'], config['name_col']]} ({n['distance']:.1f}m)" for n in others])

                result_record = {
                    f'{config["id_col"]}_Gốc': df.loc[p_original_index, config['id_col']], f'{config["name_col"]}_Gốc': df.loc[p_original_index, config['name_col']], f'{config["route_col"]}_Gốc': p_route,
                    'Hàng_xóm_gần_nhất_ID': df.loc[closest['q_original_index'], config['id_col']], 'Hàng_xóm_gần_nhất_Tên': df.loc[closest['q_original_index'], config['name_col']], 'Hàng_xóm_gần_nhất_Tuyến': df.loc[closest['q_original_index'], config['route_col']],
                    'Khoảng_cách(m)': f"{closest['distance']:.1f}", 'Các_hàng_xóm_khác_tuyến_khác': others_str,
                    'lat_goc': df.loc[p_original_index, config['lat_col']], 'lon_goc': df.loc[p_original_index, config['lon_col']],
                    'lat_hang_xom': df.loc[closest['q_original_index'], config['lat_col']], 'lon_hang_xom': df.loc[closest['q_original_index'], config['lon_col']],
                }
                for col in config.get('extra_cols', []):
                    result_record[col] = df.loc[p_original_index, col]
                results_list.append(result_record)
        
        status_callback(f"Phân tích hoàn tất. Tìm thấy {len(results_list)} điểm có sự phân tuyến bất hợp lý.")
        progress_callback(85, "Đang tạo file kết quả...")
        if not results_list:
            return "NO_RESULT", None

        results_df = pd.DataFrame(results_list)
        excel_buffer, map_object = None, None

        if config['export_excel']:
            status_callback("Đang tạo file Excel...")
            excel_buffer = io.BytesIO()
            cols_to_export = [col for col in results_df.columns if not col.startswith(('lat_', 'lon_'))]
            results_df[cols_to_export].to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_buffer.seek(0)

        if config['export_map']:
            status_callback("Đang tạo bản đồ trực quan...")
            map_object = create_analysis_map(df_valid, results_df, config)

        end_time = time.time()
        status_callback(f"Hoàn thành! Tổng thời gian: {end_time - start_time:.2f} giây.")
        progress_callback(100, "Hoàn thành!")
        return excel_buffer, map_object

    except Exception as e:
        status_callback(f"LỖI: {e}", is_error=True)
        st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")
        return None, None

def run_npp_distance_logic(config, status_callback, progress_callback):
    """
    Đo khoảng cách từ mỗi điểm bán đến trung tâm NPP tương ứng (phiên bản Streamlit).
    """
    try:
        start_time = time.time()
        status_callback("Bắt đầu quá trình đo khoảng cách NPP...")
        
        # Lấy cấu hình
        npp_id_col = config['npp_id_col']
        name_col = config['name_col']
        lat_col = config['lat_col']
        lon_col = config['lon_col']

        # Đọc file
        status_callback(f"Đang đọc file: {config['file_name']}...")
        df = pd.read_excel(config['file_data'], dtype=str)

        # Kiểm tra cột
        required_cols = [npp_id_col, name_col, lat_col, lon_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: raise ValueError(f"Các cột sau không tồn tại trong file: {', '.join(missing_cols)}")

        # Chuẩn bị và xác thực dữ liệu
        status_callback("Đang chuẩn bị và xác thực tọa độ...")
        progress_callback(10, "Đang xác thực dữ liệu...")
        df['is_valid'] = True
        df, nan_count = clean_lat_lon(df, lat_col=lat_col, lon_col=lon_col)
        if nan_count > 0: status_callback(f"Đã đánh dấu {nan_count} dòng có lat/long không hợp lệ.")
        df, invalid_range_count = validate_lat_lon(df, lat_col=lat_col, lon_col=lon_col)
        if invalid_range_count > 0: status_callback(f"Đã đánh dấu {invalid_range_count} dòng có lat/long ngoài khoảng.")
        
        df_valid = df[df['is_valid']].copy()
        status_callback(f"Tìm thấy {len(df_valid)}/{len(df)} điểm có vị trí hợp lệ.")
        if df_valid.empty: raise ValueError("Không có dữ liệu hợp lệ để xử lý.")

        # Tách điểm trung tâm NPP và điểm bán hàng
        status_callback("Tách điểm trung tâm NPP và điểm bán hàng...")
        progress_callback(30, "Đang xác định các trung tâm...")
        is_center = df_valid[name_col].str.startswith('TC_', na=False)
        df_centers_raw = df_valid[is_center].copy()
        if df_centers_raw.empty: raise ValueError("Không tìm thấy điểm trung tâm nào (có tên bắt đầu bằng 'TC_').")

        df_centers = df_centers_raw.drop_duplicates(subset=[npp_id_col], keep='first').copy()
        status_callback(f"Đã xác định {len(df_centers)} trung tâm NPP duy nhất.")

        status_callback("Ghép cặp điểm bán hàng với trung tâm NPP tương ứng...")
        lat_map = df_centers.set_index(npp_id_col)[lat_col]
        lon_map = df_centers.set_index(npp_id_col)[lon_col]
        df_valid['lat_npp'] = df_valid[npp_id_col].map(lat_map)
        df_valid['lon_npp'] = df_valid[npp_id_col].map(lon_map)

        status_callback("Tính toán khoảng cách hàng loạt (vector hóa)...")
        progress_callback(60, "Đang tính toán khoảng cách...")
        calculable_mask = (~is_center) & (df_valid['lat_npp'].notna())
        if calculable_mask.any():
            calculable_rows = df_valid[calculable_mask]
            lat1_rad, lon1_rad = np.radians(calculable_rows[lat_col].values.astype(float)), np.radians(calculable_rows[lon_col].values.astype(float))
            lat2_rad, lon2_rad = np.radians(calculable_rows['lat_npp'].values.astype(float)), np.radians(calculable_rows['lon_npp'].values.astype(float))
            dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
            distances_m = EARTH_RADIUS_METERS * (2 * np.arcsin(np.sqrt(a)))
            df_valid.loc[calculable_mask, 'KhoangCach_den_NPP(m)'] = distances_m.round(2)

        status_callback("Chuẩn bị file kết quả...")
        progress_callback(90, "Đang tạo file Excel...")
        if 'KhoangCach_den_NPP(m)' in df_valid.columns: df['KhoangCach_den_NPP(m)'] = df_valid['KhoangCach_den_NPP(m)']
        output_buffer = io.BytesIO()
        df.drop(columns=['is_valid'], errors='ignore').to_excel(output_buffer, index=False, engine='xlsxwriter')
        output_buffer.seek(0)
        end_time = time.time()
        status_callback(f"Hoàn thành! Tổng thời gian: {end_time - start_time:.2f} giây.")
        progress_callback(100, "Hoàn thành!")
        return output_buffer
    except Exception as e:
        status_callback(f"LỖI: {e}", is_error=True)
        st.error(f"Đã xảy ra lỗi trong quá trình đo đạc: {e}")
        return None

def run_duplicate_detection_logic(config, status_callback, progress_callback):
    """
    Tìm các điểm bán có nghi vấn trùng lặp trong cùng một file (phiên bản Streamlit).
    """
    try:
        start_time = time.time()
        status_callback("Bắt đầu quá trình tìm điểm trùng lặp...")

        # 1. Đọc và kiểm tra file
        status_callback(f"Đang đọc file: {config['file_name']}...")
        df = pd.read_excel(config['file_data'], dtype=str)
        df.reset_index(inplace=True) # Thêm cột 'index' để giữ lại ID dòng gốc

        required_cols = [config['lat_col'], config['lon_col']]
        if config.get('enable_name_matching'):
            required_cols.append(config['name_col'])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Các cột sau không tồn tại trong file: {', '.join(missing_cols)}")

        # 2. Chuẩn bị và xác thực dữ liệu
        status_callback("Đang chuẩn bị và xác thực tọa độ...")
        df['is_valid'] = True
        df, nan_count = clean_lat_lon(df, lat_col=config['lat_col'], lon_col=config['lon_col'])
        if nan_count > 0: status_callback(f"Đã đánh dấu {nan_count} dòng có lat/long không hợp lệ.")
        df, invalid_range_count = validate_lat_lon(df, lat_col=config['lat_col'], lon_col=config['lon_col'])
        if invalid_range_count > 0: status_callback(f"Đã đánh dấu {invalid_range_count} dòng có lat/long ngoài khoảng.")

        df_valid = df[df['is_valid']].copy()
        status_callback(f"Tìm thấy {len(df_valid)}/{len(df)} điểm có vị trí hợp lệ để phân tích.")
        if df_valid.empty: raise ValueError("Không có dữ liệu hợp lệ để phân tích.")

        # 3. Xây dựng BallTree và Truy vấn
        status_callback("Xây dựng cây dữ liệu không gian...")
        coords = np.radians(df_valid[[config['lat_col'], config['lon_col']]].values.astype(float))
        tree = BallTree(coords, metric='haversine')

        radius_rad = config['radius_m'] / EARTH_RADIUS_METERS
        status_callback(f"Truy vấn các điểm trong bán kính {config['radius_m']}m...")
        indices, distances = tree.query_radius(coords, r=radius_rad, return_distance=True, sort_results=True)

        # 4. Phân tích và lọc kết quả
        status_callback("Phân tích và lọc các cặp điểm trùng lặp...")
        progress_callback(10, "Đang phân tích cặp điểm...")
        
        if config.get('enable_name_matching'):
            status_callback("Chuẩn hóa tên để so khớp...")
            abbreviation_map = config.get('abbreviation_map', {})
            df_valid['normalized_name'] = df_valid[config['name_col']].apply(
                lambda x: normalize_name(x, abbreviation_map)
            )

        potential_pairs = []
        for i in range(len(df_valid)):
            p_iloc = i
            p_original_index = df_valid.iloc[p_iloc]['index']
            for j_idx, neighbor_iloc in enumerate(indices[p_iloc]):
                if p_iloc >= neighbor_iloc: continue
                q_original_index = df_valid.iloc[neighbor_iloc]['index']
                distance_m = distances[p_iloc][j_idx] * EARTH_RADIUS_METERS
                score = 100.0
                if config.get('enable_name_matching'):
                    name1 = df_valid.iloc[p_iloc]['normalized_name']
                    name2 = df_valid.iloc[neighbor_iloc]['normalized_name']
                    score = fuzz.token_sort_ratio(name1, name2)
                if score >= config.get('name_match_threshold', 0):
                    potential_pairs.append({'index_1': p_original_index, 'index_2': q_original_index, 'distance_m': distance_m, 'name_score': score})
        
        progress_callback(40, "Đã tạo các cặp tiềm năng...")
        if not potential_pairs:
            status_callback("Hoàn thành! Không tìm thấy cặp điểm nào thỏa mãn điều kiện.")
            return "NO_RESULT"

        status_callback(f"Tìm thấy {len(potential_pairs)} cặp điểm tiềm năng. Đang xử lý kết quả...")
        pairs_df = pd.DataFrame(potential_pairs)

        # 5. Xử lý tùy chọn kết quả
        if config['result_option'] == 'nearest_only' and not pairs_df.empty:
            status_callback("Lọc để chỉ lấy cặp gần nhất cho mỗi nhóm...")
            G = nx.Graph()
            G.add_nodes_from(df['index'].unique())
            for _, row in pairs_df.iterrows():
                G.add_edge(row['index_1'], row['index_2'], weight=row['distance_m'])
            final_pairs = []
            for component in nx.connected_components(G):
                if len(component) > 1:
                    subgraph = G.subgraph(component)
                    min_edge = min(subgraph.edges(data=True), key=lambda x: x[2]['weight'])
                    final_pairs.append((min_edge[0], min_edge[1]))
            final_indices = set(map(lambda x: tuple(sorted(x)), final_pairs))
            pairs_df['key'] = pairs_df.apply(lambda row: tuple(sorted((row['index_1'], row['index_2']))), axis=1)
            pairs_df = pairs_df[pairs_df['key'].isin(final_indices)].drop(columns=['key'])

        # 6. Chuẩn bị và xuất file kết quả
        status_callback("Chuẩn bị file kết quả...")
        progress_callback(70, "Đang tạo nhóm và ghép dữ liệu...")
        G = nx.Graph(); G.add_edges_from(pairs_df[['index_1', 'index_2']].values)
        group_info = [{'index': idx, 'ID_Nhom_Trung': f"Nhom_{i+1}"} for i, comp in enumerate(nx.connected_components(G)) if len(comp) > 1 for idx in comp]
        group_df = pd.DataFrame(group_info)
        pairs_forward = pairs_df.rename(columns={'index_1': 'index', 'index_2': 'ID_Doi_Trung'})
        pairs_backward = pairs_df.rename(columns={'index_2': 'index', 'index_1': 'ID_Doi_Trung'})
        all_pairs_info = pd.concat([pairs_forward, pairs_backward], ignore_index=True)
        all_pairs_info.rename(columns={'distance_m': 'Khoang_Cach_Trung(m)', 'name_score': 'Do_Khop_Ten_Trung(%)'}, inplace=True)
        extra_cols = config.get('extra_cols', [])
        if extra_cols:
            df_extra_info = df[['index'] + extra_cols].copy()
            df_extra_info.columns = ['ID_Doi_Trung'] + [f"{col}_DoiTrung" for col in extra_cols]
            all_pairs_info = pd.merge(all_pairs_info, df_extra_info, on='ID_Doi_Trung', how='left')
        result_df = df.copy()
        if not group_df.empty: result_df = pd.merge(result_df, group_df, on='index', how='left')
        cols_to_merge = ['index', 'ID_Doi_Trung', 'Khoang_Cach_Trung(m)', 'Do_Khop_Ten_Trung(%)'] + [f"{col}_DoiTrung" for col in extra_cols]
        if not all_pairs_info.empty: result_df = pd.merge(result_df, all_pairs_info[cols_to_merge], on='index', how='left')
        
        status_callback("Đang tạo file Excel để tải về...")
        progress_callback(95, "Đang tạo file Excel...")
        output_buffer = io.BytesIO()
        result_df.drop(columns=['index', 'is_valid']).to_excel(output_buffer, index=False, engine='xlsxwriter')
        output_buffer.seek(0)
        end_time = time.time()
        status_callback(f"Hoàn thành! Tìm thấy {len(pairs_df)} cặp. Tổng thời gian: {end_time - start_time:.2f} giây.")
        progress_callback(100, "Hoàn thành!")
        return output_buffer
    except Exception as e:
        status_callback(f"LỖI: {e}", is_error=True)
        st.error(f"Đã xảy ra lỗi trong quá trình quét: {e}")
        return None

# =============================================================================
# GIAO DIỆN STREAMLIT
# =============================================================================

st.set_page_config(page_title="VÀI CÔNG CỤ CỦA GTM", layout="wide")

# 1. Script AdSense thường đặt trong <head>
adsense_head = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-4683040057330545"
     crossorigin="anonymous"></script>
"""

# Nhúng script vào DOM (height=0 để không chiếm chỗ)
st.components.v1.html(adsense_head, height=0)

# 2. Vùng hiển thị quảng cáo
ad_code = """
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-4683040057330545"
     data-ad-slot="1234567890"
     data-ad-format="auto"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
"""

# Nhúng quảng cáo vào ứng dụng
st.components.v1.html(ad_code, height=250)


st.title("SOME OF GTM TOOLKIT")
st.write("Web Version - Xây dựng bởi Trịnh Đức Hiếu & Gemini")

# --- Khởi tạo Session State để lưu trữ dữ liệu ---
if 'abbreviation_dict' not in st.session_state:
    st.session_state.abbreviation_dict = {
        "cong ty co phan": "", "trach nhiem huu han": "", "tap hoa": "",
        "sieu thi": "", "cua hang": "", "hieu thuoc": ""
    }
if 'file1_cols' not in st.session_state: st.session_state.file1_cols = []
if 'file2_cols' not in st.session_state: st.session_state.file2_cols = []
if 'route_file_cols' not in st.session_state: st.session_state.route_file_cols = []
if 'npp_file_cols' not in st.session_state: st.session_state.npp_file_cols = []
if 'dup_file_cols' not in st.session_state: st.session_state.dup_file_cols = []

# --- Giao diện chính ---
tab1, tab2, tab3, tab4 = st.tabs([
    "  Tìm điểm lân cận  ", 
    "  Phân tích Tuyến Bán hàng  ", 
    "  Đo khoảng cách NPP  ", 
    "  Tìm điểm trùng lặp  "
])

with tab1:
    st.header("So khớp và tìm điểm trùng từ File Universal")

    # --- Cột chính và cột phụ ---
    main_col, options_col = st.columns(2)

    with main_col:
        st.subheader("1. Tải lên các file Excel")
        uploaded_file_1 = st.file_uploader("Chọn File chứa điểm - Universal", type="xlsx")
        uploaded_file_2 = st.file_uploader("Chọn File gốc để so sánh (cần đưa kết quả vào)", type="xlsx")

        # Đọc cột khi file được tải lên
        if uploaded_file_1:
            try:
                st.session_state.file1_cols = pd.read_excel(uploaded_file_1, nrows=0).columns.tolist()
            except Exception as e:
                st.error(f"Lỗi đọc file gốc: {e}")
        if uploaded_file_2:
            try:
                st.session_state.file2_cols = pd.read_excel(uploaded_file_2, nrows=0).columns.tolist()
            except Exception as e:
                st.error(f"Lỗi đọc file so sánh: {e}")

        st.subheader("2. Thiết lập các cột tọa độ")
        lat_col = st.selectbox("Cột Vĩ độ (Latitude)", st.session_state.file1_cols, index=st.session_state.file1_cols.index('latitude') if 'latitude' in st.session_state.file1_cols else 0)
        lon_col = st.selectbox("Cột Kinh độ (Longitude)", st.session_state.file1_cols, index=st.session_state.file1_cols.index('longitude') if 'longitude' in st.session_state.file1_cols else 0)
        distance_m = st.number_input("Khoảng cách (mét)", min_value=1, value=50)
        read_as_text = st.checkbox("Đọc tất cả dữ liệu dạng văn bản (giữ số 0 ở đầu)", value=True)

    with options_col:
        st.subheader("3. Tùy chọn nâng cao")
        
        # So khớp tên
        with st.expander("So khớp tên", expanded=False):
            enable_name_matching = st.checkbox("Bật so khớp tên")
            name_col1 = st.selectbox("Cột tên file gốc", st.session_state.file1_cols, index=st.session_state.file1_cols.index('Tên KH') if 'Tên KH' in st.session_state.file1_cols else 0, disabled=not enable_name_matching)
            name_col2 = st.selectbox("Cột tên file so sánh", st.session_state.file2_cols, index=st.session_state.file2_cols.index('Tên KH') if 'Tên KH' in st.session_state.file2_cols else 0, disabled=not enable_name_matching)
            name_match_threshold = st.slider("Ngưỡng khớp tên (%)", 0, 100, 85, disabled=not enable_name_matching)

        # Từ điển
        with st.expander("Quản lý từ điển viết tắt", expanded=False):
            st.write("Các từ trong 'Từ gốc' sẽ được thay thế bằng 'Từ thay thế' trước khi so sánh.")
            edited_dict_df = st.data_editor(
                pd.DataFrame(list(st.session_state.abbreviation_dict.items()), columns=['Từ gốc (không dấu)', 'Từ thay thế']),
                num_rows="dynamic",
                use_container_width=True
            )
            # Cập nhật lại session state khi người dùng chỉnh sửa
            st.session_state.abbreviation_dict = dict(zip(edited_dict_df['Từ gốc (không dấu)'], edited_dict_df['Từ thay thế']))

        # Ánh xạ cột
        with st.expander("Tùy chỉnh cột cần lấy từ file Universal", expanded=True):
            default_mapping_text = ""
            DEFAULT_COLUMN_MAPPING = {
                'KH_TUONG_UNG': 'Mã KH', 'TÊN KH_TUONG UNG': 'Tên KH',
                'lat2': 'latitude', 'long2': 'longitude'
            }
            for new_col, source_col in DEFAULT_COLUMN_MAPPING.items():
                default_mapping_text += f"{new_col}, {source_col}\n"
            
            mapping_text = st.text_area("Định dạng: tên_cột_mới, tên_cột_gốc", value=default_mapping_text, height=150)

    st.divider()

    # --- Nút thực thi và khu vực hiển thị kết quả ---
    st.subheader("4. Thực thi và Tải kết quả")
    if st.button("Bắt đầu xử lý", type="primary", use_container_width=True):
        if not uploaded_file_1 or not uploaded_file_2:
            st.error("Vui lòng tải lên đầy đủ 2 file.")
        else:
            # Parse column mapping
            parsed_map = {}
            for line in mapping_text.strip().split('\n'):
                if ',' in line:
                    parts = [p.strip() for p in line.split(',', 1)]
                    if len(parts) == 2 and parts[0] and parts[1]:
                        parsed_map[parts[0]] = parts[1]
            
            if not parsed_map:
                st.error("Ánh xạ cột không hợp lệ hoặc bị rỗng.")
            else:
                # Chuẩn bị config
                config = {
                    'file1_data': uploaded_file_1, 'file1_name': uploaded_file_1.name,
                    'file2_data': uploaded_file_2, 'file2_name': uploaded_file_2.name,
                    'distance_m': distance_m,
                    'lat_col': lat_col, 'lon_col': lon_col,
                    'column_mapping': parsed_map,
                    'read_as_text': read_as_text,
                    'enable_name_matching': enable_name_matching,
                    'abbreviation_map': dict(sorted(st.session_state.abbreviation_dict.items(), key=lambda item: len(item[0]), reverse=True))
                }
                if enable_name_matching:
                    config.update({
                        'name_match_threshold': name_match_threshold,
                        'name_col1': name_col1,
                        'name_col2': name_col2,
                    })

                # Khu vực hiển thị log và progress bar
                progress_bar = st.progress(0, text="Đang chờ...")
                log_area = st.empty()
                
                def status_callback(message, is_error=False):
                    if is_error:
                        log_area.error(message)
                    else:
                        log_area.info(message)

                def progress_callback(percent, text):
                    progress_bar.progress(percent, text=text)

                # Chạy logic
                result_buffer = run_processing_logic(config, status_callback, progress_callback)

                # Hiển thị nút tải về
                if result_buffer:
                    st.success("Xử lý hoàn tất! Bạn có thể tải file kết quả về.")
                    output_filename = f"{os.path.splitext(uploaded_file_2.name)[0]}_ketqua.xlsx"
                    st.download_button(
                        label="Tải file kết quả",
                        data=result_buffer,
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

with tab4:
    st.header("Tìm các điểm bán có nghi vấn trùng lặp")

    dup_main_col, dup_options_col = st.columns(2)

    with dup_main_col:
        st.subheader("1. Tải lên file Excel")
        uploaded_dup_file = st.file_uploader("Chọn File dữ liệu cần quét", type="xlsx", key="dup_uploader")

        if uploaded_dup_file:
            try:
                st.session_state.dup_file_cols = pd.read_excel(uploaded_dup_file, nrows=0).columns.tolist()
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")

        st.subheader("2. Thiết lập các cột và tham số")
        dup_lat_col = st.selectbox("Cột Vĩ độ (Latitude)", st.session_state.dup_file_cols, index=st.session_state.dup_file_cols.index('latitude') if 'latitude' in st.session_state.dup_file_cols else 0, key="dup_lat")
        dup_lon_col = st.selectbox("Cột Kinh độ (Longitude)", st.session_state.dup_file_cols, index=st.session_state.dup_file_cols.index('longitude') if 'longitude' in st.session_state.dup_file_cols else 0, key="dup_lon")
        dup_radius_m = st.number_input("Bán kính quét (mét)", min_value=1, value=50, key="dup_radius")

    with dup_options_col:
        st.subheader("3. Tùy chọn nâng cao")

        with st.expander("So khớp tên", expanded=False):
            dup_enable_name_matching = st.checkbox("Bật so khớp tên", key="dup_name_match_check")
            dup_name_col = st.selectbox("Cột Tên điểm bán", st.session_state.dup_file_cols, index=st.session_state.dup_file_cols.index('Tên KH') if 'Tên KH' in st.session_state.dup_file_cols else 0, disabled=not dup_enable_name_matching, key="dup_name_col")
            dup_name_match_threshold = st.slider("Ngưỡng khớp tên (%)", 0, 100, 85, disabled=not dup_enable_name_matching, key="dup_threshold")

        with st.expander("Tùy chọn kết quả", expanded=True):
            dup_result_option = st.radio(
                "Lựa chọn kết quả:",
                ('Lấy tất cả các cặp điểm thỏa mãn', 'Chỉ lấy cặp gần nhất trong mỗi nhóm trùng'),
                key="dup_result_option"
            )
            dup_extra_cols = st.multiselect(
                "Chọn các cột thông tin thêm để hiển thị cho điểm đối ứng:",
                st.session_state.dup_file_cols,
                key="dup_extra_cols"
            )

    st.divider()

    st.subheader("4. Thực thi và Tải kết quả")
    if st.button("Bắt đầu quét", type="primary", use_container_width=True, key="dup_run_button"):
        if not uploaded_dup_file:
            st.error("Vui lòng tải lên File dữ liệu cần quét.")
        else:
            # Chuẩn bị config
            config = {
                'file_data': uploaded_dup_file, 'file_name': uploaded_dup_file.name,
                'radius_m': dup_radius_m,
                'lat_col': dup_lat_col, 'lon_col': dup_lon_col,
                'result_option': 'nearest_only' if dup_result_option == 'Chỉ lấy cặp gần nhất trong mỗi nhóm trùng' else 'all',
                'extra_cols': dup_extra_cols,
                'enable_name_matching': dup_enable_name_matching,
                'abbreviation_map': dict(sorted(st.session_state.abbreviation_dict.items(), key=lambda item: len(item[0]), reverse=True))
            }
            if dup_enable_name_matching:
                if not dup_name_col:
                    st.error("Khi bật so khớp tên, vui lòng chọn Cột Tên điểm bán.")
                else:
                    config.update({
                        'name_match_threshold': dup_name_match_threshold,
                        'name_col': dup_name_col,
                    })
            
            # Chỉ chạy nếu không có lỗi ở trên
            if not (dup_enable_name_matching and not dup_name_col):
                progress_bar_dup = st.progress(0, text="Đang chờ...")
                log_area_dup = st.empty()

                def status_callback_dup(message, is_error=False):
                    if is_error: log_area_dup.error(message)
                    else: log_area_dup.info(message)

                def progress_callback_dup(percent, text):
                    progress_bar_dup.progress(percent, text=text)

                # Chạy logic
                result_buffer_dup = run_duplicate_detection_logic(config, status_callback_dup, progress_callback_dup)

                if result_buffer_dup == "NO_RESULT":
                    st.warning("Không tìm thấy cặp điểm nào thỏa mãn điều kiện.")
                elif result_buffer_dup:
                    st.success("Quét hoàn tất! Bạn có thể tải file kết quả về.")
                    output_filename_dup = f"{os.path.splitext(uploaded_dup_file.name)[0]}_trunglap_ketqua.xlsx"
                    st.download_button(
                        label="Tải file kết quả quét trùng lặp",
                        data=result_buffer_dup,
                        file_name=output_filename_dup,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dup_download_button"
                    )

with tab3:
    st.header("Đo khoảng cách từ Điểm bán về Trung tâm NPP")

    npp_main_col, npp_options_col = st.columns([2, 1])

    with npp_main_col:
        st.subheader("1. Tải lên & Thiết lập Cột")
        uploaded_npp_file = st.file_uploader("Chọn File Dữ liệu (chứa cả Điểm bán và Trung tâm)", type="xlsx", key="npp_uploader")

        if uploaded_npp_file:
            try:
                st.session_state.npp_file_cols = pd.read_excel(uploaded_npp_file, nrows=0).columns.tolist()
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
        
        c1, c2 = st.columns(2)
        with c1:
            npp_lat_col = st.selectbox("Cột Vĩ độ", st.session_state.npp_file_cols, index=st.session_state.npp_file_cols.index('latitude') if 'latitude' in st.session_state.npp_file_cols else 0, key="npp_lat")
            npp_id_col = st.selectbox("Cột Mã NPP", st.session_state.npp_file_cols, index=st.session_state.npp_file_cols.index('Mã NPP') if 'Mã NPP' in st.session_state.npp_file_cols else 0, key="npp_id")
        with c2:
            npp_lon_col = st.selectbox("Cột Kinh độ", st.session_state.npp_file_cols, index=st.session_state.npp_file_cols.index('longitude') if 'longitude' in st.session_state.npp_file_cols else 0, key="npp_lon")
            npp_name_col = st.selectbox("Cột Tên Điểm bán", st.session_state.npp_file_cols, index=st.session_state.npp_file_cols.index('Tên KH') if 'Tên KH' in st.session_state.npp_file_cols else 0, key="npp_name")

    with npp_options_col:
        st.subheader("2. Hướng dẫn")
        st.info(
            "Tính năng này sẽ tính khoảng cách từ mỗi điểm bán đến trung tâm NPP tương ứng trong cùng một file.\n\n"
            "**Yêu cầu quan trọng:**\n"
            "1. Các điểm **Trung tâm NPP** phải có tên trong Cột Tên Điểm bán bắt đầu bằng `TC_` (ví dụ: `TC_NPP Hà Nội`).\n"
            "2. Cả điểm bán và trung tâm phải có cùng giá trị trong **Cột Mã NPP** để được ghép cặp."
        )

    st.divider()

    st.subheader("3. Thực thi và Tải kết quả")
    if st.button("Bắt đầu đo đạc", type="primary", use_container_width=True, key="npp_run_button"):
        if not uploaded_npp_file:
            st.error("Vui lòng tải lên File Dữ liệu.")
        else:
            config = {
                'file_data': uploaded_npp_file, 'file_name': uploaded_npp_file.name,
                'npp_id_col': npp_id_col, 'name_col': npp_name_col,
                'lat_col': npp_lat_col, 'lon_col': npp_lon_col,
            }

            progress_bar_npp = st.progress(0, text="Đang chờ...")
            log_area_npp = st.empty()

            def status_callback_npp(message, is_error=False):
                if is_error: log_area_npp.error(message)
                else: log_area_npp.info(message)

            def progress_callback_npp(percent, text):
                progress_bar_npp.progress(percent, text=text)

            result_buffer_npp = run_npp_distance_logic(config, status_callback_npp, progress_callback_npp)

            if result_buffer_npp:
                st.success("Đo đạc hoàn tất! Bạn có thể tải file kết quả về.")
                output_filename_npp = f"{os.path.splitext(uploaded_npp_file.name)[0]}_kc_npp_ketqua.xlsx"
                st.download_button(
                    label="Tải file kết quả đo đạc",
                    data=result_buffer_npp,
                    file_name=output_filename_npp,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="npp_download_button"
                )

with tab2:
    st.header("Phân tích sự bất hợp lý trong phân tuyến")

    if not FOLIUM_AVAILABLE:
        st.error("Tính năng này yêu cầu thư viện 'folium'. Vui lòng chạy lệnh: pip install folium")
    else:
        route_main_col, route_options_col = st.columns([2, 1]) # Cột chính rộng hơn

        with route_main_col:
            st.subheader("1. Tải lên & Thiết lập Cột")
            uploaded_route_file = st.file_uploader("Chọn File Phân tuyến", type="xlsx", key="route_uploader")

            if uploaded_route_file:
                try:
                    st.session_state.route_file_cols = pd.read_excel(uploaded_route_file, nrows=0).columns.tolist()
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
            
            c1, c2 = st.columns(2)
            with c1:
                route_lat_col = st.selectbox("Cột Vĩ độ", st.session_state.route_file_cols, index=st.session_state.route_file_cols.index('latitude') if 'latitude' in st.session_state.route_file_cols else 0, key="route_lat")
                route_day_col = st.selectbox("Cột Phân tuyến", st.session_state.route_file_cols, index=st.session_state.route_file_cols.index('Thứ') if 'Thứ' in st.session_state.route_file_cols else 0, key="route_day")
                route_name_col = st.selectbox("Cột Tên hiển thị", st.session_state.route_file_cols, index=st.session_state.route_file_cols.index('Tên KH') if 'Tên KH' in st.session_state.route_file_cols else 0, key="route_name")
            with c2:
                route_lon_col = st.selectbox("Cột Kinh độ", st.session_state.route_file_cols, index=st.session_state.route_file_cols.index('longitude') if 'longitude' in st.session_state.route_file_cols else 0, key="route_lon")
                route_id_col = st.selectbox("Cột Định danh", st.session_state.route_file_cols, index=st.session_state.route_file_cols.index('Mã KH') if 'Mã KH' in st.session_state.route_file_cols else 0, key="route_id")

        with route_options_col:
            st.subheader("2. Tham số & Tùy chọn")
            route_dist_m = st.number_input("Ngưỡng khoảng cách (mét)", min_value=1, value=50, key="route_dist")
            
            with st.expander("Lọc và Hiển thị", expanded=True):
                route_filter_text = st.text_input("Lọc theo tuyến (cách nhau bởi dấu phẩy, trống = tất cả)", key="route_filter")
                route_tooltip_cols = st.multiselect("Cột Hover (Tooltip) trên bản đồ", st.session_state.route_file_cols, default=['Tên KH'] if 'Tên KH' in st.session_state.route_file_cols else None, key="route_tooltip")
                route_extra_cols = st.multiselect("Cột Popup (Thông tin thêm)", st.session_state.route_file_cols, key="route_extra")

            st.subheader("3. Kết quả đầu ra")
            route_export_excel = st.checkbox("Xuất file Excel", value=True, key="route_excel")
            route_export_map = st.checkbox("Tạo bản đồ trực quan (HTML)", value=True, key="route_map")

        st.divider()

        st.subheader("4. Thực thi và Tải kết quả")
        if st.button("Bắt đầu phân tích", type="primary", use_container_width=True, key="route_run_button"):
            if not uploaded_route_file:
                st.error("Vui lòng tải lên File Phân tuyến.")
            else:
                filter_routes_list = [r.strip() for r in route_filter_text.split(',') if r.strip()]
                
                config = {
                    'file_data': uploaded_route_file, 'file_name': uploaded_route_file.name,
                    'lat_col': route_lat_col, 'lon_col': route_lon_col,
                    'route_col': route_day_col, 'id_col': route_id_col, 'name_col': route_name_col,
                    'distance_m': route_dist_m,
                    'export_excel': route_export_excel, 'export_map': route_export_map,
                    'filter_routes': filter_routes_list,
                    'tooltip_cols': route_tooltip_cols, 'extra_cols': route_extra_cols,
                }

                progress_bar_route = st.progress(0, text="Đang chờ...")
                log_area_route = st.empty()

                def status_callback_route(message, is_error=False):
                    if is_error: log_area_route.error(message)
                    else: log_area_route.info(message)

                def progress_callback_route(percent, text):
                    progress_bar_route.progress(percent, text=text)

                excel_buffer, map_object = run_route_analysis_logic(config, status_callback_route, progress_callback_route)

                if excel_buffer == "NO_RESULT":
                    st.warning("Không tìm thấy điểm nào có sự phân tuyến bất hợp lý với các điều kiện đã cho.")
                else:
                    dl_col1, dl_col2 = st.columns(2)
                    if excel_buffer:
                        with dl_col1:
                            st.download_button(
                                label="Tải file Excel kết quả",
                                data=excel_buffer,
                                file_name=f"{os.path.splitext(uploaded_route_file.name)[0]}_phantuyen_ketqua.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    if map_object:
                        # Chuyển đổi map thành HTML
                        map_html_bytes = map_object._repr_html_().encode()
                        with dl_col2:
                            st.download_button(
                                label="Tải file Bản đồ (HTML)",
                                data=map_html_bytes,
                                file_name=f"{os.path.splitext(uploaded_route_file.name)[0]}_bando.html",
                                mime="text/html",
                                use_container_width=True
                            )
                        
                        # Hiển thị bản đồ
                        st.subheader("Bản đồ trực quan")

                        st.components.v1.html(map_html_bytes, height=600, scrolling=True)




