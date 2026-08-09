using System;
using System.Collections;
using System.Collections.Generic;

using KitchenSafety.Models;
using KitchenSafety.UI;

using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace KitchenSafety.Networking
{
    /// <summary>
    /// Uploads one imported test image to the FastAPI
    /// mock-analysis endpoint.
    /// </summary>
    public class TestImageUploader : MonoBehaviour
    {
        [Header("Backend")]

        [SerializeField]
        private BackendClient backendClient;


        [Header("Test Image")]

        [SerializeField]
        private Texture2D testImage;

        [SerializeField]
        [Range(30, 100)]
        private int jpegQuality = 75;


        [Header("Interface")]

        [SerializeField]
        private Button uploadTestImageButton;

        [SerializeField]
        private TMP_Text analysisStatusText;

        [SerializeField]
        private HazardDisplay hazardDisplay;


        [Header("Request")]

        [SerializeField]
        [Min(1)]
        private int timeoutSeconds = 30;


        private bool uploadInProgress;


        private void Awake()
        {
            if (uploadTestImageButton != null)
            {
                uploadTestImageButton.onClick.AddListener(
                    UploadTestImage
                );
            }
        }


        private void OnDestroy()
        {
            if (uploadTestImageButton != null)
            {
                uploadTestImageButton.onClick.RemoveListener(
                    UploadTestImage
                );
            }
        }


        /// <summary>
        /// Called when the Analyse Test Image button is pressed.
        /// </summary>
        public void UploadTestImage()
        {
            if (uploadInProgress)
            {
                Debug.LogWarning(
                    "An image upload is already in progress."
                );
                return;
            }

            if (backendClient == null)
            {
                SetStatus(
                    "Backend Client is not assigned."
                );
                return;
            }

            if (testImage == null)
            {
                SetStatus(
                    "Test image is not assigned."
                );
                return;
            }

            StartCoroutine(UploadImageCoroutine());
        }


        private IEnumerator UploadImageCoroutine()
        {
            uploadInProgress = true;

            if (uploadTestImageButton != null)
            {
                uploadTestImageButton.interactable = false;
            }

            hazardDisplay?.HideHazard();

            SetStatus("Encoding test image…");

            byte[] jpegBytes;

            try
            {
                jpegBytes = ConvertTextureToJpeg(
                    sourceTexture: testImage,
                    quality: jpegQuality
                );
            }
            catch (Exception exception)
            {
                Debug.LogError(
                    $"Image encoding failed: " +
                    $"{exception.Message}\n" +
                    $"{exception.StackTrace}"
                );

                SetStatus(
                    "Could not encode the test image."
                );

                FinishUpload();
                yield break;
            }

            // Do not create a multipart file section unless
            // valid bytes were produced.
            if (jpegBytes == null || jpegBytes.Length == 0)
            {
                Debug.LogError(
                    "JPEG encoding returned no image data."
                );

                SetStatus(
                    "Image encoding produced no data."
                );

                FinishUpload();
                yield break;
            }

            Debug.Log(
                $"Test image encoded successfully. " +
                $"JPEG size: {jpegBytes.Length} bytes."
            );

            string captureId =
                $"TEST_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}";

            List<IMultipartFormSection> formSections =
                new List<IMultipartFormSection>
                {
                    new MultipartFormDataSection(
                        "capture_id",
                        captureId
                    ),

                    new MultipartFormFileSection(
                        "image",
                        jpegBytes,
                        "unity_test_image.jpg",
                        "image/jpeg"
                    )
                };

            string endpointUrl = backendClient.BuildUrl(
                "/api/v1/test/analyse-image"
            );

            SetStatus("Uploading test image…");

            using UnityWebRequest request =
                UnityWebRequest.Post(
                    endpointUrl,
                    formSections
                );

            request.timeout = timeoutSeconds;

            yield return request.SendWebRequest();

            if (
                request.result
                != UnityWebRequest.Result.Success
            )
            {
                string responseBody =
                    request.downloadHandler != null
                        ? request.downloadHandler.text
                        : "";

                Debug.LogError(
                    $"Test image upload failed.\n" +
                    $"URL: {endpointUrl}\n" +
                    $"HTTP status: {request.responseCode}\n" +
                    $"Error: {request.error}\n" +
                    $"Response: {responseBody}"
                );

                SetStatus(
                    $"Upload failed: {request.error}"
                );

                FinishUpload();
                yield break;
            }

            string responseJson =
                request.downloadHandler.text;

            Debug.Log(
                $"Mock hazard response:\n{responseJson}"
            );

            HazardAnalysisResponse response;

            try
            {
                response =
                    JsonUtility.FromJson<
                        HazardAnalysisResponse
                    >(responseJson);
            }
            catch (Exception exception)
            {
                Debug.LogError(
                    $"Could not parse hazard JSON: " +
                    $"{exception.Message}\n" +
                    $"Response: {responseJson}"
                );

                SetStatus(
                    "Backend returned invalid JSON."
                );

                FinishUpload();
                yield break;
            }

            if (response == null)
            {
                SetStatus(
                    "Backend returned an empty response."
                );

                FinishUpload();
                yield break;
            }

            if (response.capture_id != captureId)
            {
                Debug.LogWarning(
                    $"Capture ID mismatch. " +
                    $"Sent: {captureId}; " +
                    $"received: {response.capture_id}"
                );

                SetStatus(
                    "Response belongs to another capture."
                );

                FinishUpload();
                yield break;
            }

            if (
                response.hazards_present
                && response.hazards != null
                && response.hazards.Length > 0
            )
            {
                SetStatus(
                    $"Analysis complete: " +
                    $"{response.hazards.Length} " +
                    $"hazard(s)"
                );
            }
            else
            {
                SetStatus(
                    "Analysis complete: no hazards"
                );
            }

            hazardDisplay?.DisplayResponse(response);

            FinishUpload();
        }


        /// <summary>
        /// Converts any GPU texture into a readable,
        /// uncompressed RGB24 texture and then encodes it
        /// as JPEG.
        ///
        /// This works even when the original imported texture
        /// uses a compressed texture format.
        /// </summary>
        private byte[] ConvertTextureToJpeg(
            Texture sourceTexture,
            int quality
        )
        {
            if (sourceTexture == null)
            {
                throw new ArgumentNullException(
                    nameof(sourceTexture)
                );
            }

            if (
                sourceTexture.width <= 0
                || sourceTexture.height <= 0
            )
            {
                throw new InvalidOperationException(
                    "The source texture has invalid dimensions."
                );
            }

            RenderTexture temporaryRenderTexture =
                RenderTexture.GetTemporary(
                    sourceTexture.width,
                    sourceTexture.height,
                    0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Default
                );

            RenderTexture previousRenderTexture =
                RenderTexture.active;

            Texture2D readableTexture = null;

            try
            {
                // Copy the compressed or GPU-only texture to an
                // uncompressed render target.
                Graphics.Blit(
                    sourceTexture,
                    temporaryRenderTexture
                );

                RenderTexture.active =
                    temporaryRenderTexture;

                // RGB24 is directly supported by JPEG encoding.
                readableTexture = new Texture2D(
                    sourceTexture.width,
                    sourceTexture.height,
                    TextureFormat.RGB24,
                    mipChain: false
                );

                readableTexture.ReadPixels(
                    new Rect(
                        0,
                        0,
                        temporaryRenderTexture.width,
                        temporaryRenderTexture.height
                    ),
                    destX: 0,
                    destY: 0,
                    recalculateMipMaps: false
                );

                readableTexture.Apply(
                    updateMipmaps: false,
                    makeNoLongerReadable: false
                );

                byte[] encodedBytes =
                    ImageConversion.EncodeToJPG(
                        readableTexture,
                        quality
                    );

                if (
                    encodedBytes == null
                    || encodedBytes.Length == 0
                )
                {
                    throw new InvalidOperationException(
                        "Unity returned no JPEG data."
                    );
                }

                return encodedBytes;
            }
            finally
            {
                RenderTexture.active =
                    previousRenderTexture;

                RenderTexture.ReleaseTemporary(
                    temporaryRenderTexture
                );

                if (readableTexture != null)
                {
                    Destroy(readableTexture);
                }
            }
        }


        private void FinishUpload()
        {
            uploadInProgress = false;

            if (uploadTestImageButton != null)
            {
                uploadTestImageButton.interactable = true;
            }
        }


        private void SetStatus(string message)
        {
            if (analysisStatusText != null)
            {
                analysisStatusText.text = message;
            }

            Debug.Log(message);
        }
    }
}