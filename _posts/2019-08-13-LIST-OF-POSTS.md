---

layout : post
title : LIST-OF-POSTS
categories: [MAIN]
comments : true
tags : [MAIN]

---

<hr class='division4'>
<br>

<div class="toc">
  <ul class="posts">
  {% for item in site.posts %}
    <li class="text-title">
      <a href="{{ site.baseurl }}{{ item.url }}">
        {{ item.title }}
      </a>
    </li>
  {% endfor %}
  </ul>
</div>
<br><br><br>
