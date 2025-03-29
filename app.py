import streamlit as st
import pandas as pd

    sections = {
        "For You": get_fair_recommendations(data).head(10),
        "Interested": get_interest_based_recommendations(data),
        "Trending Now": data.sample(10, replace=True),
        "Most Watched": data.sample(10, replace=True),
    }

    for section, content in sections.items():
        st.subheader(section)
        cards_html = ""
        for _, row in content.iterrows():
            json_data = json.dumps(row.to_dict()).replace("\"", "&quot;")
            cards_html += f"""
                <div class='item'>
                    <img src="{row['image']}" alt="{row['title']}">
                    <div class='title'>{row['title']}</div>
                    <form method='post'>
                        <button onclick='selectBroadcast(`{json_data}`)' type='button'>View</button>
                    </form>
                </div>
            """

        section_html = f"""
        <style>
        .gallery-wrapper {{ position: relative; width: 100%; margin-bottom: 30px; }}
        .container {{ display: flex; overflow-x: auto; scroll-behavior: smooth; gap: 20px; padding: 10px 0; }}
        .item {{ flex: 0 0 auto; width: 220px; background: #f0f2f6; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; }}
        .item img {{ width: 100%; height: 130px; object-fit: cover; border-top-left-radius: 10px; border-top-right-radius: 10px; }}
        .item .title {{ font-weight: 600; padding: 8px; font-size: 14px; }}
        .item button {{ margin-bottom: 10px; background-color: #007bff; color: white; border: none; padding: 6px 12px; border-radius: 5px; cursor: pointer; }}
        .scroll-btn {{ position: absolute; top: 40%; transform: translateY(-50%); background: rgba(0, 0, 0, 0.5); color: white; border: none; padding: 10px; font-size: 20px; cursor: pointer; border-radius: 50%; z-index: 10; }}
        .scroll-left {{ left: 0; }}
        .scroll-right {{ right: 0; }}
        </style>
        <div class='gallery-wrapper'>
            <button class='scroll-btn scroll-left' onclick="scrollGallery_{section.replace(' ', '_')}(-1)">&#8249;</button>
            <div class='container' id='scrollable-gallery-{section.replace(' ', '_')}'>
                {cards_html}
            </div>
            <button class='scroll-btn scroll-right' onclick="scrollGallery_{section.replace(' ', '_')}(1)">&#8250;</button>
        </div>
        <script>
        function scrollGallery_{section.replace(' ', '_')}(direction) {{
            const container = document.getElementById("scrollable-gallery-{section.replace(' ', '_')}");
            const scrollAmount = 240;
            container.scrollLeft += direction * scrollAmount;
        }}

        function selectBroadcast(jsonStr) {{
            const form = document.createElement('form');
            form.method = 'POST';
            form.style.display = 'none';
            const input = document.createElement('input');
            input.name = 'selected_broadcast'
            input.value = jsonStr;
            form.appendChild(input);
            document.body.appendChild(form);
            form.submit();
        }}
        </script>
        """

        components.html(section_html, height=360, scrolling=False)