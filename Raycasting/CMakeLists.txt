add_executable(Raycasting Core/Core.h Core/Core.cpp Core/Vector.h Core/Input.h Core/Input.cpp Core/Camera.h Core/Camera.cpp Renderer/Renderer.h Renderer/Renderer.cpp Renderer/HostRenderer.h Renderer/HostRenderer.cpp Renderer/DeviceRenderer.h Renderer/DeviceRenderer.cu Renderer/Shaders.h Renderer/Shaders.cu Models.h Scene.h Scene.cpp Window.h Window.cpp main.cpp)

target_include_directories(Raycasting PRIVATE ${CMAKE_SOURCE_DIR}/Raycasting)
target_include_directories(Raycasting PRIVATE ${CMAKE_SOURCE_DIR}/vendor/imgui/imgui/backends)
target_link_libraries(Raycasting glfw glad imgui glm)