# Mobile view of jupyter notebook html
## Step 01:
```bash
# We need classic template converted html (instead of default download)
jupyter nbconvert --to html --template classic your_notebook.ipynb
```

## Step 02:
Make sure we have this in <head> section.
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

## Step 03: Make sure we have this css style

```html
pre, code, table {
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}
table {
    width: 100%;
}
```


Example:
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    body {
        width: 100%;
        margin: 0;
        padding: 0;
    }
    pre, code, table {
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    table {
        width: 100%;
    }
    .container {
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
    }
</style>
```

# Mobile view for already converted html files

## Step 01: file: assets/css/style_mobile.css
```css
/* styles/css/style_mobile.css */
body {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: 16px !important;
}

pre, code, table, .output, .input {
    overflow-x: auto !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

table {
    width: 100% !important;
    max-width: 100% !important;
}

/* Force images to scale down */
img {
    max-width: 100% !important;
    height: auto !important;
}

/* Fix long lines in output cells */
.output_text, .input_area {
    width: 100% !important;
}

/* Adjust padding for mobile */
.container {
    padding: 10px !important;
}
```

## Step 02: Add styles to html file
```html
<head>
    <!-- Existing meta tags, title, etc. -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="../styles/css/style_mobile.css">
</head>
```

## Step 03: Alternative method of direct inline css

```html
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Paste the entire style_mobile.css content here */
        body { width: 100% !important; margin: 0 !important; ... }
    </style>
</head>
```