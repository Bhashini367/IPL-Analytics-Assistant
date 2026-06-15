import fitz

import os


def extract_images(
    pdf_path,
    output_dir="images"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    pdf = fitz.open(pdf_path)

    image_paths = []

    for page_index in range(len(pdf)):

        page = pdf[page_index]

        images = page.get_images(
            full=True
        )

        for img_idx, img in enumerate(images):

            xref = img[0]

            base_image = pdf.extract_image(
                xref
            )

            image_bytes = (
                base_image["image"]
            )

            image_path = (
                f"{output_dir}/"
                f"page_{page_index+1}_"
                f"{img_idx}.png"
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    image_bytes
                )

            image_paths.append(
                image_path
            )

    pdf.close()

    return image_paths