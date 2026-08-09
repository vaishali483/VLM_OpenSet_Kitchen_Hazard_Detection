using System;
using System.Collections;
using System.Collections.Generic;

using KitchenSafety.Models;
using KitchenSafety.Networking;
using KitchenSafety.UI;
using KitchenSafety.Sessions;

using TMPro;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace KitchenSafety.Capture
{
    /// <summary>
    /// Captures a frame from the AR Foundation device camera,
    /// converts it to JPEG, uploads it to FastAPI and displays
    /// the returned hazard result.
    ///
    /// The backend endpoint still returns a mock hazard during
    /// this development stage.
    /// </summary>
    public class ARCameraFrameUploader : MonoBehaviour
    {
        public enum ImageRotation
        {
            None,
            Clockwise90,
            Clockwise180,
            Clockwise270
        }


        [Header("AR Camera")]

        [SerializeField]
        private ARCameraManager cameraManager;


        [Header("Backend")]

        [SerializeField]
        private BackendClient backendClient;

        [Header("Cooking Session")]

        [SerializeField]
        private CookingSessionController cookingSessionController;


        [Header("Image Encoding")]

        [Tooltip(
            "Maximum width before optional rotation. " +
            "The image is downsampled to reduce upload size."
        )]
        [SerializeField]
        [Range(320, 1280)]
        private int maximumOutputWidth = 640;

        [SerializeField]
        [Range(30, 100)]
        private int jpegQuality = 70;

        [Tooltip(
            "Mirror across the image X axis. In AR Foundation, " +
            "this visually flips the image vertically."
        )]
        [SerializeField]
        private bool flipVertically = true;

        [Tooltip(
            "Mirror across the image Y axis. In AR Foundation, " +
            "this visually flips the image horizontally."
        )]
        [SerializeField]
        private bool flipHorizontally = false;

        [SerializeField]
        private ImageRotation imageRotation =
            ImageRotation.Clockwise90;


        [Header("Interface")]

        [SerializeField]
        private Button captureFrameButton;

        [SerializeField]
        private TMP_Text captureStatusText;

        [SerializeField]
        private HazardDisplay hazardDisplay;

        [SerializeField]
        private RawImage lastCapturePreview;

        [SerializeField]
        private AspectRatioFitter previewAspectRatioFitter;


        [Header("Request")]

        [SerializeField]
        [Min(1)]
        private int timeoutSeconds = 30;


        private bool requestInProgress;
        private Texture2D lastCapturedTexture;

        public bool RequestInProgress
        {
            get
            {
                return requestInProgress;
            }
        }


        /// <summary>
        /// True when the required camera and backend references exist.
        /// </summary>
        public bool IsReady
        {
            get
            {
                return cameraManager != null
                    && backendClient != null;
            }
        }


        private void Awake()
        {
            if (captureFrameButton != null)
            {
                captureFrameButton.onClick.AddListener(
                    CaptureAndUpload
                );
            }
        }


        private void OnDestroy()
        {
            if (captureFrameButton != null)
            {
                captureFrameButton.onClick.RemoveListener(
                    CaptureAndUpload
                );
            }

            if (lastCapturedTexture != null)
            {
                Destroy(lastCapturedTexture);
            }
        }


        /// <summary>
        /// Called by the manual Analyse Now button.
        /// </summary>
        public void CaptureAndUpload()
        {
            TryStartCaptureAndUpload();
        }


        /// <summary>
        /// Attempts to start one AR capture and upload.
        ///
        /// Returns true only when a camera image was acquired and
        /// the upload coroutine was started.
        /// </summary>
        public bool TryStartCaptureAndUpload()
        {
            if (requestInProgress)
            {
                SetStatus(
                    "An analysis request is already running."
                );

                return false;
            }

            if (cameraManager == null)
            {
                SetStatus(
                    "AR Camera Manager is not assigned."
                );

                return false;
            }

            if (backendClient == null)
            {
                SetStatus(
                    "Backend Client is not assigned."
                );

                return false;
            }

            if (
                cookingSessionController != null
                && cookingSessionController.HasOpenSession
                && !cookingSessionController.HasActiveSession
            )
            {
                SetStatus(
                    "Cooking session is paused."
                );

                return false;
            }

            if (
                !cameraManager.TryAcquireLatestCpuImage(
                    out XRCpuImage cpuImage
                )
            )
            {
                SetStatus(
                    "Camera frame is not ready. Trying again…"
                );

                return false;
            }

            // The existing warning text remains visible.
            // Only localisation belonging to the previous image is cleared.
            hazardDisplay?.NotifyCaptureStarted();

            StartCoroutine(
                CaptureAndUploadCoroutine(cpuImage)
            );

            return true;
        }


        private IEnumerator CaptureAndUploadCoroutine(
            XRCpuImage cpuImage
        )
        {
            requestInProgress = true;

            if (captureFrameButton != null)
            {
                captureFrameButton.interactable = false;
            }

            SetStatus("Converting AR camera frame…");

            byte[] jpegBytes;

            try
            {
                jpegBytes = ConvertCpuImageToJpeg(
                    cpuImage
                );
            }
            catch (Exception exception)
            {
                Debug.LogError(
                    $"AR camera conversion failed:\n" +
                    $"{exception.Message}\n" +
                    $"{exception.StackTrace}"
                );

                SetStatus(
                    "Could not convert the AR camera frame."
                );

                FinishRequest();
                yield break;
            }
            finally
            {
                // XRCpuImage owns a limited native camera
                // resource and must always be disposed.
                cpuImage.Dispose();
            }

            if (
                jpegBytes == null
                || jpegBytes.Length == 0
            )
            {
                SetStatus(
                    "Camera conversion produced no JPEG data."
                );

                FinishRequest();
                yield break;
            }

            Debug.Log(
                $"AR frame encoded successfully. " +
                $"JPEG size: {jpegBytes.Length} bytes."
            );

            string captureId =
                $"AR_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}";

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
                        "ar_camera_frame.jpg",
                        "image/jpeg"
                    )
                };

            string relativeEndpoint =
                cookingSessionController != null
                && cookingSessionController.HasActiveSession
                    ? cookingSessionController
                        .GetAnalysisEndpoint()
                    : "/api/v1/live/analyse-image";

            string endpointUrl =
                backendClient.BuildUrl(
                    relativeEndpoint
                );

            SetStatus("Uploading and analysing kitchen scene…");

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
                    $"AR frame upload failed.\n" +
                    $"URL: {endpointUrl}\n" +
                    $"HTTP status: {request.responseCode}\n" +
                    $"Error: {request.error}\n" +
                    $"Response: {responseBody}"
                );

                SetStatus(
                    $"Upload failed: {request.error}"
                );

                FinishRequest();
                yield break;
            }

            string responseJson =
                request.downloadHandler.text;

            Debug.Log(
                $"AR camera mock response:\n" +
                $"{responseJson}"
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
                    $"Could not parse hazard JSON:\n" +
                    $"{exception.Message}\n" +
                    $"Response: {responseJson}"
                );

                SetStatus(
                    "Backend returned invalid JSON."
                );

                FinishRequest();
                yield break;
            }

            if (response == null)
            {
                SetStatus(
                    "Backend returned an empty response."
                );

                FinishRequest();
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
                    "Received a result for another capture."
                );

                FinishRequest();
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

            FinishRequest();
        }


        /// <summary>
        /// Converts an AR Foundation XRCpuImage into a resized
        /// and correctly oriented JPEG.
        /// </summary>
        private byte[] ConvertCpuImageToJpeg(
            XRCpuImage cpuImage
        )
        {
            if (
                !cpuImage.FormatSupported(
                    TextureFormat.RGB24
                )
            )
            {
                throw new InvalidOperationException(
                    "This device does not support RGB24 " +
                    "CPU image conversion."
                );
            }

            int outputWidth = Mathf.Min(
                maximumOutputWidth,
                cpuImage.width
            );

            float scale =
                outputWidth / (float)cpuImage.width;

            int outputHeight = Mathf.Max(
                1,
                Mathf.RoundToInt(
                    cpuImage.height * scale
                )
            );

            Vector2Int outputDimensions =
                new Vector2Int(
                    outputWidth,
                    outputHeight
                );

            XRCpuImage.Transformation transformation =
                XRCpuImage.Transformation.None;

            if (flipVertically)
            {
                transformation |=
                    XRCpuImage.Transformation.MirrorX;
            }

            if (flipHorizontally)
            {
                transformation |=
                    XRCpuImage.Transformation.MirrorY;
            }

            XRCpuImage.ConversionParams conversionParams =
                new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(
                        0,
                        0,
                        cpuImage.width,
                        cpuImage.height
                    ),

                    outputDimensions = outputDimensions,
                    outputFormat = TextureFormat.RGB24,
                    transformation = transformation
                };

            Texture2D convertedTexture =
                new Texture2D(
                    outputWidth,
                    outputHeight,
                    TextureFormat.RGB24,
                    mipChain: false
                );

            try
            {
                NativeArray<byte> rawTextureData =
                    convertedTexture
                        .GetRawTextureData<byte>();

                NativeSlice<byte> destinationBuffer =
                    new NativeSlice<byte>(
                        rawTextureData
                    );

                cpuImage.Convert(
                    conversionParams,
                    destinationBuffer
                );

                convertedTexture.Apply(
                    updateMipmaps: false,
                    makeNoLongerReadable: false
                );

                Texture2D orientedTexture =
                    ApplyRotation(
                        convertedTexture,
                        imageRotation
                    );

                if (
                    orientedTexture
                    != convertedTexture
                )
                {
                    Destroy(convertedTexture);
                }

                ReplaceLastCapturedTexture(
                    orientedTexture
                );

                byte[] encodedBytes =
                    ImageConversion.EncodeToJPG(
                        lastCapturedTexture,
                        jpegQuality
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
            catch
            {
                if (
                    convertedTexture != null
                    && convertedTexture
                       != lastCapturedTexture
                )
                {
                    Destroy(convertedTexture);
                }

                throw;
            }
        }


        private void ReplaceLastCapturedTexture(
            Texture2D newTexture
        )
        {
            if (
                lastCapturedTexture != null
                && lastCapturedTexture != newTexture
            )
            {
                Destroy(lastCapturedTexture);
            }

            lastCapturedTexture = newTexture;

            if (lastCapturePreview != null)
            {
                lastCapturePreview.texture =
                    lastCapturedTexture;
            }

            if (
                previewAspectRatioFitter != null
                && lastCapturedTexture.height > 0
            )
            {
                previewAspectRatioFitter.aspectRatio =
                    lastCapturedTexture.width
                    / (float)lastCapturedTexture.height;
            }
        }


        private static Texture2D ApplyRotation(
            Texture2D source,
            ImageRotation rotation
        )
        {
            if (rotation == ImageRotation.None)
            {
                return source;
            }

            Color32[] sourcePixels =
                source.GetPixels32();

            int sourceWidth = source.width;
            int sourceHeight = source.height;

            int destinationWidth;
            int destinationHeight;

            if (
                rotation == ImageRotation.Clockwise90
                || rotation
                   == ImageRotation.Clockwise270
            )
            {
                destinationWidth = sourceHeight;
                destinationHeight = sourceWidth;
            }
            else
            {
                destinationWidth = sourceWidth;
                destinationHeight = sourceHeight;
            }

            Color32[] destinationPixels =
                new Color32[sourcePixels.Length];

            for (
                int sourceY = 0;
                sourceY < sourceHeight;
                sourceY++
            )
            {
                for (
                    int sourceX = 0;
                    sourceX < sourceWidth;
                    sourceX++
                )
                {
                    int sourceIndex =
                        sourceY * sourceWidth
                        + sourceX;

                    int destinationX;
                    int destinationY;

                    switch (rotation)
                    {
                        case ImageRotation.Clockwise90:
                            destinationX =
                                sourceHeight - 1 - sourceY;

                            destinationY = sourceX;
                            break;

                        case ImageRotation.Clockwise180:
                            destinationX =
                                sourceWidth - 1 - sourceX;

                            destinationY =
                                sourceHeight - 1 - sourceY;
                            break;

                        case ImageRotation.Clockwise270:
                            destinationX = sourceY;

                            destinationY =
                                sourceWidth - 1 - sourceX;
                            break;

                        default:
                            destinationX = sourceX;
                            destinationY = sourceY;
                            break;
                    }

                    int destinationIndex =
                        destinationY * destinationWidth
                        + destinationX;

                    destinationPixels[
                        destinationIndex
                    ] = sourcePixels[sourceIndex];
                }
            }

            Texture2D rotatedTexture =
                new Texture2D(
                    destinationWidth,
                    destinationHeight,
                    TextureFormat.RGB24,
                    mipChain: false
                );

            rotatedTexture.SetPixels32(
                destinationPixels
            );

            rotatedTexture.Apply(
                updateMipmaps: false,
                makeNoLongerReadable: false
            );

            return rotatedTexture;
        }


        private void FinishRequest()
        {
            requestInProgress = false;

            if (captureFrameButton != null)
            {
                captureFrameButton.interactable = true;
            }
        }


        private void SetStatus(string message)
        {
            if (captureStatusText != null)
            {
                captureStatusText.text = message;
            }

            Debug.Log(message);
        }
    }
}