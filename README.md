<div align="center">
<img width="120px" height="120px" src="public/sample.png" alt="SpineXR">

# SpineXR Detection

_Abnormalities classification and detection for spinal lesions in radiographs_

[![Star](public/logo.png)](https://github.com/vinbigdata-medical/vindr-spinexr)

</div>

The aim of this project is to develop a high-quality video annotation tool for computer vision and machine learning
applications with the following desiderata:

1. Simple and efficient to use for a non-expert.
2. Supports multiple annotation types including temporal segments, object bounding boxes, semantic and instance regions,
   tracklets, and human pose (skeleton).
3. Runs in a browser without external libraries or need for server-side processing. But easy to plug-in a back-end for
   heavy "in-the-loop" processing (e.g., segments from bounding boxes or frame completion from partial labels).
4. Integrates easily with crowd-sourced annotation services (e.g., Amazon Mechanical Turk).
5. Compatible with all (most) modern browsers and operating systems including tablets.
6. Secure. Data does not need to leave the local machine (since there is no server-side processing).
7. Open-source.

## Screenshots

### Object

![Object](doc/img/object.gif)

### Region

![Region](doc/img/region.gif)

### Skeleton

![Skeleton](doc/img/skeleton.gif)

### Skeleton Type

![Skeleton Type](doc/img/skeleton-type.gif)

### Action

![Action](doc/img/action.gif)

## Usage

### Annotate local videos

Just open [Host](https://vidat2.davidz.cn) and open a local video, you are good to go!

### Annotate remote videos

You need to [deploy](#deployment) Vidat first, and then use [URL parameters](#url-parameters) to load the video into
Vidat. Please note that Vidat does **not** support online YouTube videos due
to [Cross-Origin Resource Sharing (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS).

### Integrate with Amazon Mechanical Turk (MTurk)

1. **Prepare tasks**
    1. [Deploy](#deployment) Vidat on a server which can access to the videos and annotation (config) files.
    2. Generate URLs for each task,
       e.g. `https://example.com?annotation=task1.json&submitURL=http%3A%2F%2Fexample2.com%3Ftoken%3D123456`.
2. **Dispatch tasks on MTurk**
    1. Create a new MTurk task with survey template, replace the survey link with task link.
    2. Create a batch with generated URLs.
3. **Collect submissions**
    1. Build up an independent API backend (see `/tools/backend/` for a simple implementation) that handles submissions.

Submission API:

**Request**

```text
POST <submitURL>
content-type: application/json
<annotation>
```

**Respond**

```text
content-type: application/json
{
    type: '' // color: "primary" (default) | "secondary" | "accent" | "dark" | "positive" | "negative" | "info" | "warning"
    message: '' // notify the user (required)
    clipboard: '' // copy to user's clipboard (optional)
}
```

## Deployment

> Note that this is only necessary if you want to do development or host your own version of the tool. If you just want to label videos then you can use one of the host servers linked to above (data will remain on your local machine; it will not be sent to the host server).

1. Download our latest [release](https://github.com/anucvml/vidat/releases). Note that the `pre-release` is
   automatically generated and should **not** be used in production.
2. Unzip all files and put them behind a web server ([Nginx](http://nginx.org/), [Apache](http://httpd.apache.org/),
   etc.). Note that open `index.html` in your explorer does **not** work.
3. Open in your favourite browser.

## URL Parameters

> All the keys and values are **not** case-sensitive.
>
> **Note** if you are using an external URL for `annotation`, `video`, `config` or `submitURL`,
> please make sure you are following [Cross-Origin Resource Sharing (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS).
> And they need to be [URL encoded](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURI) if there is any special characters.

### `annotation`

**Default** `null`

**Example** `/annotation/exmaple.json`, `http://exmaple/static/annotation/exmaple.json`

Path to the annotation file. Vidat will load the video, annotation and configurations from this file. This parameter has
higher priority than `video`, `config`, `defaultFPS` and `defaultFPK`. Please refer
to [File Formats - Annotation](#file-formats) for format details.

### `video`

**Default** `null`

**Example** `/video/exmaple.mp4`, `http://exmaple/static/video/exmaple.json`

Path to the video file. Please refer to [`decoder`](#decoder) for more information.

### `config`

**Default** `null`

**Example** `/config/exmaple.json`, `http://exmaple/static/config/exmaple.json`

Path to the video file. Please refer to [File Formats - Config](#file-formats) for format details.

### `mode`

**Default** `null`

**Example** `object` | `region` | `skeleton`

Specify current mode for Vidat.

### `zoom`

**Default** `false`

**Example** `true` | `false`

Whether toggle zoom on.

### `sensitivity`

**Default** `hasTouch ? 10 : 5`

**Example** `Integer >= 1`

When detecting points / edges, the number of pixel(s) between you mouse and the annotation.

### `defaultFPS`

**Default** `10`

**Example** `1 <= Integer <= 60`

The default frame per second used when extracting frames from the given video.

### `defaultFPK`

**Default** `50`

**Example** `Integer >= 1`

The default frame per keyframe used when generating keyframes.

### `decoder`

**Default** `auto`

**Example** `auto` | `v1` | `v2`

The video decoder used for frame extracting.

`v1` uses `<canvas>` as a video decoder,
by [`pause` - `draw` - `play` - `wait for timeupdate` strategy](https://stackoverflow.com/revisions/32708998/5). It is
the most reliable and compatible methods for most cases. But it is slow and computational inefficient.

`v2` uses [`WebCodecs.VideoDecoder`](https://developer.mozilla.org/en-US/docs/Web/API/VideoDecoder), it takes the
advantages of native video decoder built inside the browser. It is way faster than `v1` but lack of support from old
browsers.

`auto` Vidat will determine which one to use for you.

See [VideoLoader Wiki](https://github.com/anucvml/vidat/wiki/VideoLoader) for details.

### `showObjects`

**Default** `true`

**Example** `true` | `false`

Whether to show `object` mode related components.

### `showRegions`

**Default** `true`

**Example** `true` | `false`

Whether to show `region` mode related components.

### `showSkeletons`

**Default** `true`

**Example** `true` | `false`

Whether to show `skeleton` mode related components.

### `showActions`

**Default** `true`

**Example** `true` | `false`

Whether to show `action` related components.

### `muted`

**Default** `true`

**Example** `true` | `false`

Whether to mute the video when playing.

### `grayscale`

**Default** `false`

**Example** `true` | `false`

Whether to grayscale the video.

### `showPopup`

**Default** `true`

**Example** `true` | `false`

Whether to show quick popup when finishing annotating an object/region/skeleton.

### `submitURL`

**Default** `null`

**Example** `submitURL=http%3A%2F%2Fexample.com%3Ftoken%3D123456`

URL used for submitting annotation.

### Examples

```
http://example.com?showObjects=false&showRegions=false&showSkeletons=false
```

This will show action only.

```
http://example.com?mode=skeleton&showPopup=false
```

This will set the current mode to skeleton and disable popup window.



**Annotation**
Following COCO format:

```json
{
    "images": [
        {
            "file_name": "1f1c5c098c35b79052596aae08ac727f.png",
            "height": 2874,
            "width": 1183,
            "id": 0
        },
        {
            "file_name": "9f4fb4f51315715cf1048fcc2f7d6f47.png",
            "height": 2874,
            "width": 1021,
            "id": 1
        }
    ],
    "categories": [
        {
            "supercategory": "Defect",
            "id": 1,
            "name": "Osteophytes"
        },
        {
            "supercategory": "Defect",
            "id": 2,
            "name": "Other lesions"
        },
        {
            "supercategory": "Defect",
            "id": 3,
            "name": "Spondylolysthesis"
        },
        {
            "supercategory": "Defect",
            "id": 4,
            "name": "Disc space narrowing"
        },
        {
            "supercategory": "Defect",
            "id": 5,
            "name": "Vertebral collapse"
        },
        {
            "supercategory": "Defect",
            "id": 6,
            "name": "Foraminal stenosis"
        },
        {
            "supercategory": "Defect",
            "id": 7,
            "name": "Surgical implant"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 0,
            "bbox": [
                712.0,
                961.0,
                73.0,
                50.0
            ],
            "area": 3650,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 2,
            "image_id": 0,
            "bbox": [
                789.0,
                1452.0,
                90.0,
                53.0
            ],
            "area": 4770,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 3,
            "image_id": 0,
            "bbox": [
                692.0,
                655.0,
                53.0,
                53.0
            ],
            "area": 2809,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 4,
            "image_id": 0,
            "bbox": [
                820.0,
                1603.0,
                90.0,
                67.0
            ],
            "area": 6030,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 5,
            "image_id": 0,
            "bbox": [
                739.0,
                1203.0,
                80.0,
                36.0
            ],
            "area": 2880,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 6,
            "image_id": 0,
            "bbox": [
                860.0,
                1710.0,
                80.0,
                60.0
            ],
            "area": 4800,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 7,
            "image_id": 0,
            "bbox": [
                793.0,
                1368.0,
                63.0,
                47.0
            ],
            "area": 2961,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 8,
            "image_id": 0,
            "bbox": [
                732.0,
                1119.0,
                67.0,
                57.0
            ],
            "area": 3819,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 9,
            "image_id": 0,
            "bbox": [
                699.0,
                517.0,
                60.0,
                47.0
            ],
            "area": 2820,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 10,
            "image_id": 0,
            "bbox": [
                877.0,
                1892.0,
                73.0,
                60.0
            ],
            "area": 4380,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 11,
            "image_id": 1,
            "bbox": [
                658.0,
                601.0,
                40.0,
                53.0
            ],
            "area": 2120,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 12,
            "image_id": 1,
            "bbox": [
                635.0,
                685.0,
                70.0,
                43.0
            ],
            "area": 3010,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 13,
            "image_id": 1,
            "bbox": [
                779.0,
                1673.0,
                90.0,
                97.0
            ],
            "area": 8730,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 14,
            "image_id": 1,
            "bbox": [
                692.0,
                1166.0,
                94.0,
                63.0
            ],
            "area": 5922,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 15,
            "image_id": 1,
            "bbox": [
                746.0,
                1593.0,
                84.0,
                50.0
            ],
            "area": 4200,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 16,
            "image_id": 1,
            "bbox": [
                756.0,
                1892.0,
                77.0,
                43.0
            ],
            "area": 3311,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 17,
            "image_id": 1,
            "bbox": [
                742.0,
                1421.0,
                97.0,
                63.0
            ],
            "area": 6111,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 18,
            "image_id": 1,
            "bbox": [
                662.0,
                836.0,
                57.0,
                50.0
            ],
            "area": 2850,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 19,
            "image_id": 1,
            "bbox": [
                695.0,
                1075.0,
                60.0,
                63.0
            ],
            "area": 3780,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 20,
            "image_id": 1,
            "bbox": [
                672.0,
                917.0,
                67.0,
                63.0
            ],
            "area": 4221,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        },
        {
            "id": 21,
            "image_id": 1,
            "bbox": [
                736.0,
                1331.0,
                70.0,
                47.0
            ],
            "area": 3290,
            "iscrowd": 0,
            "category_id": 1,
            "segmentation": []
        }
    ]
}
```

See [`public/annotation/sample.json`](public/sample.json) for am example.


