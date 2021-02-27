from api.api import Api

class Get(Api):

    def info(self, api_ref, parsed_path,*args):
        return '\n'.join([
            'CLIENT VALUES:',
            'client_address=%s (%s)' % (api_ref.client_address,
                api_ref.address_string()),
            'command=%s' % api_ref.command,
            'path=%s' % api_ref.path,
            'real path=%s' % parsed_path.path,
            'query=%s' % parsed_path.query,
            'request_version=%s' % api_ref.request_version,
            '',
            'SERVER VALUES:',
            'server_version=%s' % api_ref.server_version,
            'sys_version=%s' % api_ref.sys_version,
            'protocol_version=%s' % api_ref.protocol_version,
            '',
            ])
        
    def all(self, *args):
        """Return all files currently managed by server"""
        return str(self.shared.all_files)

    def images(self, *args):
        return str(self.shared.images)

    def objects(self, *args):
        return str(self.shared.objects)

    def processes(self, *args):
        return str(self.shared.processes)

    def image(self, api_ref, data, *args):
        """Return imagefile of id specified in data"""
        return ""

    def object(self, api_ref, data, *args):
        """Return objectfile of id specified in data"""
        return ""
