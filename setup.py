from setuptools import setup, find_packages

setup(
    name="media-authenticity-verification",
    version="1.0.0",
    description="A Python system for verifying media authenticity using perceptual hashing and ML",
    author="Replit User",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "imagehash>=4.3.0",
        "moviepy>=1.0.3",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0"
    ],
    python_requires=">=3.11",
    entry_points={
        'console_scripts': [
            'media-auth-verify=app:main',
        ],
    },
)