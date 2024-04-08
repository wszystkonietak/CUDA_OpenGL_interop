#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

template <typename T>
class Surface {
public:
	Surface() = default;
	Surface(uint2 resolution, T* h_data = nullptr, unsigned int flags = cudaArraySurfaceLoadStore) : resolution(resolution)
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		cudaMallocArray(&array, &channelDesc, resolution.x, resolution.y, flags);
		cudaResourceDesc desc;
		memset(&desc, 0, sizeof(desc));

		if (h_data != nullptr)
			cudaMemcpy2DToArray(array, 0, 0, h_data, resolution.x * sizeof(T), resolution.x * sizeof(T), resolution.y, cudaMemcpyHostToDevice);

		desc.res.array.array = array;
		cudaCreateSurfaceObject(&surface, &desc);
	}
	Surface(unsigned int texture, unsigned int target, uint2 resolution) : resolution(resolution)
	{
		cudaGraphicsResource_t cudaTextureResource;
		cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, target, cudaGraphicsRegisterFlagsSurfaceLoadStore);

		cudaGraphicsMapResources(1, &cudaTextureResource, 0);
		cudaGraphicsSubResourceGetMappedArray(&array, cudaTextureResource, 0, 0);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;
		cudaCreateSurfaceObject(&surface, &resDesc);
	}
	cudaSurfaceObject_t surface = 0;
private:
	cudaArray_t array;
	uint2 resolution;
};