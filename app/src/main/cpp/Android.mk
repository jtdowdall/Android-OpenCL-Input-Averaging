LOCAL_PATH := $(call my-dir)
LOCAL_PATH_EXT := $(call my-dir)/../../CL
include $(CLEAR_VARS)

LOCAL_MODULE := native-lib

LOCAL_CFLAGS += -DANDROID_CL
LOCAL_CFLAGS += -O3 -ffast-math

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../CL

LOCAL_SRC_FILES := native-lib.cpp

#LOCAL_LDFLAGS += -ljnigraphics
LOCAL_LDLIBS := -llog -ljnigraphics
LOCAL_LDLIBS += $(LOCAL_PATH)/../../CL/libOpenCL.so

LOCAL_ARM_MODE := arm

include $(BUILD_SHARED_LIBRARY)
