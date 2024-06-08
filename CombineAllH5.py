import h5py

# 创建一个新的 HDF5 文件用于汇总
with h5py.File('trip_times.h5', 'w') as combined_file:
    for hour in range(24):
        with h5py.File(f'trip_times_hour_{hour}.h5', 'r') as hour_file:
            data = hour_file[f'hour_{hour}'][()]
            combined_file.create_dataset(f'hour_{hour}', data=data)

print("所有小时的数据已汇总到一个文件中。")
